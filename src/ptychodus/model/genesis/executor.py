from collections.abc import Callable
from pathlib import Path
import logging
import queue
import threading


from ptychodus.api.io import StandardFileLayout
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI
from ..task_manager import BackgroundTaskManager, ForegroundTask
from .iri import IRIClient, IRIComputeClient, JobSpecification

from .settings import GenesisSettings
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs
from .tasks import compute_task, transfer_task

logger = logging.getLogger(__name__)


class WorkflowTask:
    def __init__(
        self,
        transfer_client: AmSCGlobusTransferClient,
        compute_client: IRIComputeClient,
        local_to_remote_transfer_inputs: GlobusTransferInputs,
        compute_resource_id: str,
        job_specification: JobSpecification,
        remote_to_local_transfer_inputs: GlobusTransferInputs,
        stop_event: threading.Event,
        status_interval_s: float,
    ) -> None:
        self._transfer_client = transfer_client
        self._compute_client = compute_client
        self._local_to_remote_transfer_inputs = local_to_remote_transfer_inputs
        self._compute_resource_id = compute_resource_id
        self._job_specification = job_specification
        self._remote_to_local_transfer_inputs = remote_to_local_transfer_inputs
        self._stop_event = stop_event
        self._status_interval_s = status_interval_s

    def __call__(self) -> ForegroundTask | None:
        # Step 1: Transfer input data from local to remote
        logger.info('Workflow step 1: transferring input data from local to remote...')
        try:
            for status in transfer_task(
                self._transfer_client,
                self._local_to_remote_transfer_inputs,
                self._stop_event,
                self._status_interval_s,
            ):
                logger.info(f'{status.action} [{status.label}]: {status.status}')
        except Exception:
            logger.exception('Local-to-remote transfer failed!')
            return None

        if self._stop_event.is_set():
            return None

        # Step 2: Submit compute job and wait for completion
        logger.info('Workflow step 2: submitting compute job...')
        try:
            for status in compute_task(
                self._compute_client,
                self._compute_resource_id,
                self._job_specification,
                self._stop_event,
                self._status_interval_s,
            ):
                logger.info(f'{status.action} [{status.label}]: {status.status}')
        except Exception:
            logger.exception('Compute job failed!')
            return None

        if self._stop_event.is_set():
            return None

        # Step 3: Transfer output data from remote to local
        logger.info('Workflow step 3: transferring output data from remote to local...')
        try:
            for status in transfer_task(
                self._transfer_client,
                self._remote_to_local_transfer_inputs,
                self._stop_event,
                self._status_interval_s,
            ):
                logger.info(f'{status.action} [{status.label}]: {status.status}')
        except Exception:
            logger.exception('Remote-to-local transfer failed!')
            return None

        logger.info('Workflow completed successfully.')
        return None


class TaskRunner:  # FIXME finish
    def __init__(self) -> None:
        self._task_queue: queue.Queue[Callable] = queue.Queue()
        self._dependent_tasks: dict[int, list[Callable]] = dict()
        self._num_submissions = 0

    def submit(self, task: Callable, depends_on: int | None = None) -> int:
        # FIXME ensure thread-safe?
        """returns int useful for dependencies"""
        task_number = self._num_submissions

        if depends_on is None:
            self._task_queue.put(task)
        else:
            try:
                self._dependent_tasks[depends_on].append(task)
            except KeyError:
                self._dependent_tasks[depends_on] = [task]

        self._num_submissions += 1

        return task_number


class GenesisExecutor:
    def __init__(
        self,
        task_manager: BackgroundTaskManager,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
        settings: GenesisSettings,
        iri_client_chooser: PluginChooser[IRIClient],
        transfer_client_chooser: PluginChooser[AmSCGlobusTransferClient],
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._settings = settings
        self._settings_registry = settings_registry
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self._processing_api = processing_api
        self._iri_client_chooser = iri_client_chooser
        self._transfer_client_chooser = transfer_client_chooser

    def populate_input_directory(self, input_product_index: int) -> Path:
        try:
            product_item = self._product_api.get_item(input_product_index)
        except IndexError:
            logger.exception(f'Failed access product for flow ({input_product_index=})!')
            raise

        input_directory = (
            self._settings.local_collection_posix_path.get_value() / product_item.get_name()
        )

        try:
            input_directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.exception('Input data POSIX path must be a directory!')
            raise

        self._settings_registry.save_settings(input_directory / StandardFileLayout.SETTINGS)
        self._diffraction_api.export_assembled_patterns(
            input_directory / StandardFileLayout.DIFFRACTION
        )
        self._product_api.save_product(
            input_product_index,
            input_directory / StandardFileLayout.PRODUCT_IN,
            file_type='HDF5',
        )

        return input_directory

    def _run_flow(
        self,
        ptychodus_action: str,
        flow_label: str,
    ) -> None:
        transfer_client = self._transfer_client_chooser.get_current_plugin().strategy
        compute_client = self._iri_client_chooser.get_current_plugin().strategy.compute

        local_uuid = str(self._settings.local_collection_id.get_value())
        local_base = self._settings.local_collection_globus_path.get_value().rstrip('/')
        remote_uuid = str(self._settings.remote_collection_id.get_value())
        remote_base = self._settings.remote_collection_globus_path.get_value().rstrip('/')

        local_to_remote_inputs = GlobusTransferInputs(
            label=flow_label,
            source_uuid=local_uuid,
            source_path=f'{local_base}/{flow_label}',
            destination_uuid=remote_uuid,
            destination_path=f'{remote_base}/{flow_label}',
        )
        remote_to_local_inputs = GlobusTransferInputs(
            label=flow_label,
            source_uuid=remote_uuid,
            source_path=f'{remote_base}/{flow_label}',
            destination_uuid=local_uuid,
            destination_path=f'{local_base}/{flow_label}',
        )

        stop_event = threading.Event()
        status_interval_s = float(self._settings.status_refresh_interval_s.get_value())

        workflow_task = WorkflowTask(
            transfer_client=transfer_client,
            compute_client=compute_client,
            local_to_remote_transfer_inputs=local_to_remote_inputs,
            compute_resource_id=compute_resource_id,
            job_specification=job_specification,
            remote_to_local_transfer_inputs=remote_to_local_inputs,
            stop_event=stop_event,
            status_interval_s=status_interval_s,
        )
        self._task_manager.put_background_task(workflow_task)

    def reconstruct(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        input_directory = self.populate_input_directory(input_product_index)

        if self._processing_api.is_reconstructor_trainable():
            pass  # TODO get model from mlflow

        self._run_flow('reconstruct', input_directory.name)

    def train(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        input_directory = self.populate_input_directory(input_product_index)
        self._run_flow('train', input_directory.name)
        # TODO customize input/output directories; put model to mlflow
