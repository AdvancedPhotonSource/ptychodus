from datetime import datetime
from pathlib import Path
from uuid import UUID
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
from .facility_adapters import IRIFacilityAdapter
from .iri import IRIComputeClient, JobSpecification
from .settings import GenesisSettings
from .tasks import compute_task, transfer_task
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs
from .status import GenesisStatus

logger = logging.getLogger(__name__)


def combine_path_segments(base_path: str, subpath: str) -> str:
    return f'{base_path.rstrip("/")}/{subpath.lstrip("/")}'


def create_globus_url(collection_id: UUID, globus_path: str) -> str:
    return f'globus://{collection_id}/{globus_path.lstrip("/")}'


class WorkflowTask:
    def __init__(
        self,
        transfer_client: AmSCGlobusTransferClient,
        compute_client: IRIComputeClient,
        outbound_transfer_inputs: GlobusTransferInputs,
        compute_resource_id: str,
        job_specification: JobSpecification,
        inbound_transfer_inputs: GlobusTransferInputs,
        stop_event: threading.Event,
        status_q: queue.Queue[GenesisStatus],
        status_interval_s: float,
        load_product: bool,
        label: str,
        product_api: ProductAPI,
        output_product_path: Path,
    ) -> None:
        self._transfer_client = transfer_client
        self._compute_client = compute_client
        self._local_to_remote_transfer_inputs = outbound_transfer_inputs
        self._compute_resource_id = compute_resource_id
        self._job_specification = job_specification
        self._remote_to_local_transfer_inputs = inbound_transfer_inputs
        self._stop_event = stop_event
        self._status_q = status_q
        self._status_interval_s = status_interval_s
        self._load_product = load_product
        self._label = label
        self._product_api = product_api
        self._output_product_path = output_product_path
        self._start_time = datetime.now()
        self._status_q.put(
            GenesisStatus(
                label=self._label,
                start_time=self._start_time,
                completion_time=None,
                status='Waiting',
                action='Workflow',
            )
        )

    def __call__(self) -> ForegroundTask | None:
        self._status_q.put(
            GenesisStatus(
                label=self._label,
                start_time=self._start_time,
                completion_time=None,
                status='Running',
                action='Workflow',
            )
        )

        # Step 1: Transfer input data from local to remote
        logger.info('Workflow step 1: transferring input data from local to remote...')
        try:
            for status in transfer_task(
                self._transfer_client,
                self._local_to_remote_transfer_inputs,
                self._stop_event,
                self._status_interval_s,
            ):
                self._status_q.put(status)
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
                self._status_q.put(status)
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
                self._status_q.put(status)
        except Exception:
            logger.exception('Remote-to-local transfer failed!')
            return None

        result: ForegroundTask | None = None

        # Step 4: Load data product (reconstruct only)
        if self._load_product:
            logger.info('Workflow step 4: loading reconstructed product...')

            def load_result() -> None:  # FIXME clean up
                self._product_api.open_product(self._output_product_path, file_type='HDF5')

            result = load_result

        self._status_q.put(
            GenesisStatus(
                label=self._label,
                start_time=self._start_time,
                completion_time=datetime.now(),
                status='Completed',
                action='Workflow',
            )
        )

        logger.info('Workflow completed successfully.')
        return result


class GenesisExecutor:
    def __init__(
        self,
        task_manager: BackgroundTaskManager,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
        settings: GenesisSettings,
        facility_chooser: PluginChooser[IRIFacilityAdapter],
        transfer_client_chooser: PluginChooser[AmSCGlobusTransferClient],
        status_q: queue.Queue[GenesisStatus],
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._settings = settings
        self._settings_registry = settings_registry
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self._processing_api = processing_api
        self._facility_chooser = facility_chooser
        self._transfer_client_chooser = transfer_client_chooser
        self._status_q = status_q

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

        if self._processing_api.is_reconstructor_trainable():
            pass  # FIXME save model

        return input_directory

    def _run_flow(self, ptychodus_action: str, flow_label: str, *, load_product: bool) -> None:
        transfer_client = self._transfer_client_chooser.get_current_plugin().strategy
        facility_adapter = self._facility_chooser.get_current_plugin().strategy
        iri_client = facility_adapter.get_iri_client()
        compute_resource_id = self._settings.compute_resource_id.get_value()

        input_path_segment = f'{flow_label}/input'
        output_path_segment = f'{flow_label}/output'

        lc_id = self._settings.local_collection_id.get_value()
        lc_globus_path = self._settings.local_collection_globus_path.get_value()

        rc_id = self._settings.remote_collection_id.get_value()
        rc_globus_path = self._settings.remote_collection_globus_path.get_value()
        rc_posix_path = str(self._settings.remote_collection_posix_path.get_value())

        outbound_source_path = combine_path_segments(lc_globus_path, input_path_segment)
        outbound_destination_path = combine_path_segments(rc_globus_path, input_path_segment)
        inbound_source_path = combine_path_segments(rc_globus_path, output_path_segment)
        inbound_destination_path = combine_path_segments(lc_globus_path, output_path_segment)

        outbound_transfer_inputs = GlobusTransferInputs(
            source_url=create_globus_url(lc_id, outbound_source_path),
            destination_url=create_globus_url(rc_id, outbound_destination_path),
            label=f'ptychodus_{flow_label}_outbound',
            source_uuid=str(lc_id),
            source_path=outbound_source_path,
            destination_uuid=str(rc_id),
            destination_path=outbound_destination_path,
        )
        job_specification = facility_adapter.create_job_specification(
            action=ptychodus_action,
            input_directory=Path(combine_path_segments(rc_posix_path, input_path_segment)),
            output_directory=Path(combine_path_segments(rc_posix_path, output_path_segment)),
        )
        inbound_transfer_inputs = GlobusTransferInputs(
            source_url=create_globus_url(rc_id, inbound_source_path),
            destination_url=create_globus_url(lc_id, inbound_destination_path),
            label=f'ptychodus_{flow_label}_inbound',
            source_uuid=str(rc_id),
            source_path=inbound_source_path,
            destination_uuid=str(lc_id),
            destination_path=inbound_destination_path,
        )

        stop_event = threading.Event()
        status_interval_s = float(self._settings.status_refresh_interval_s.get_value())
        output_product_path = (
            self._settings.local_collection_posix_path.get_value()
            / flow_label
            / 'output'
            / StandardFileLayout.PRODUCT_OUT
        )

        workflow_task = WorkflowTask(
            transfer_client=transfer_client,
            compute_client=iri_client.compute,
            outbound_transfer_inputs=outbound_transfer_inputs,
            compute_resource_id=compute_resource_id,
            job_specification=job_specification,
            inbound_transfer_inputs=inbound_transfer_inputs,
            stop_event=stop_event,
            status_q=self._status_q,
            status_interval_s=status_interval_s,
            load_product=load_product,
            label=flow_label,
            product_api=self._product_api,
            output_product_path=output_product_path,
        )
        self._task_manager.put_background_task(workflow_task)

    def reconstruct(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)

        if self._processing_api.is_reconstructor_trainable():
            pass  # TODO get model from mlflow

        input_directory = self.populate_input_directory(input_product_index)
        self._run_flow('reconstruct', input_directory.name, load_product=True)

    def train(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        input_directory = self.populate_input_directory(input_product_index)
        self._run_flow('train', input_directory.name, load_product=False)
        # TODO customize input/output directories; put model to mlflow
