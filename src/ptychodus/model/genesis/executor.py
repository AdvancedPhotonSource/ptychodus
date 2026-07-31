from dataclasses import dataclass
from pathlib import Path
from uuid import UUID
import logging
import queue
import threading

from ptychodus.api.io import StandardFileLayout
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..processing import ProcessingAPI
from ..product import ProductAPI
from ..task_manager import BackgroundTaskManager
from .facility_adapters import IRIFacilityAdapter
from .settings import GenesisSettings
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs
from .status import GenesisStatus
from .workflow import PtychodusWorkflow

logger = logging.getLogger(__name__)


def create_globus_url(collection_id: UUID, *path_segments: str) -> str:
    path = '/'.join(segment.strip('/') for segment in path_segments)
    return f'globus://{collection_id}/{path}'


@dataclass(frozen=True)
class WorkflowDirectoryStructure:
    base_directory: Path

    @property
    def label(self) -> str:
        return self.base_directory.name

    @property
    def input_directory(self) -> Path:
        return self.base_directory / 'input'

    @property
    def input_path_segments(self) -> tuple[str, str]:
        parts = self.input_directory.parts
        return (parts[-2], parts[-1])

    @property
    def output_directory(self) -> Path:
        return self.base_directory / 'output'

    @property
    def output_path_segments(self) -> tuple[str, str]:
        parts = self.output_directory.parts
        return (parts[-2], parts[-1])


class GenesisExecutor:
    def __init__(
        self,
        task_manager: BackgroundTaskManager,
        settings_registry: SettingsRegistry,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
        settings: GenesisSettings,
        facility_chooser: PluginChooser[IRIFacilityAdapter],
        transfer_client_chooser: PluginChooser[AmSCGlobusTransferClient],
        status_q: queue.Queue[GenesisStatus],
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._settings = settings
        self._settings_registry = settings_registry
        self._product_api = product_api
        self._processing_api = processing_api
        self._facility_chooser = facility_chooser
        self._transfer_client_chooser = transfer_client_chooser
        self._status_q = status_q
        self._stop_event = stop_event

    def populate_input_directory(self, input_product_index: int) -> WorkflowDirectoryStructure:
        try:
            product_item = self._product_api.get_item(input_product_index)
        except IndexError:
            logger.exception(f'Failed access product for flow ({input_product_index=})!')
            raise

        dataset = product_item.get_dataset()

        if dataset is None:
            raise RuntimeError(
                f'Product "{product_item.get_name()}" has no associated diffraction dataset.'
            )

        local_dir_struct = WorkflowDirectoryStructure(
            self._settings.local_collection_posix_path.get_value() / product_item.get_name()
        )

        try:
            local_dir_struct.input_directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.exception('Input data POSIX path must be a directory!')
            raise

        self._settings_registry.save_settings(
            local_dir_struct.input_directory / StandardFileLayout.SETTINGS
        )
        dataset.export_assembled_patterns(
            local_dir_struct.input_directory / StandardFileLayout.DIFFRACTION
        )
        self._product_api.save_product(
            input_product_index,
            local_dir_struct.input_directory / StandardFileLayout.PRODUCT_IN,
            file_type='HDF5',
        )

        return local_dir_struct

    def _run_flow(
        self, ptychodus_action: str, local_dir_struct: WorkflowDirectoryStructure
    ) -> None:
        flow_label = f'ptychodus_{ptychodus_action}_{local_dir_struct.label}'
        transfer_client = self._transfer_client_chooser.get_current_plugin().strategy
        facility_plugin = self._facility_chooser.get_current_plugin()
        facility_adapter = facility_plugin.strategy
        facility_name = facility_plugin.display_name
        iri_client = facility_adapter.get_iri_client()
        compute_resource_id = self._settings.compute_resource_id.get_value()

        lc_id = self._settings.local_collection_id.get_value()
        lc_globus_path = self._settings.local_collection_globus_path.get_value()

        rc_id = self._settings.remote_collection_id.get_value()
        rc_globus_path = self._settings.remote_collection_globus_path.get_value()

        outbound_transfer_inputs = GlobusTransferInputs(
            source_url=create_globus_url(
                lc_id, lc_globus_path, *local_dir_struct.input_path_segments
            ),
            destination_url=create_globus_url(
                rc_id, rc_globus_path, *local_dir_struct.input_path_segments
            ),
            label='Transfer Outbound',
        )
        logger.debug(f'Created outbound transfer inputs: {outbound_transfer_inputs}')

        remote_dir_struct = WorkflowDirectoryStructure(
            self._settings.remote_collection_posix_path.get_value() / local_dir_struct.label
        )
        job_specification = facility_adapter.create_job_specification(
            action=ptychodus_action,
            input_directory=remote_dir_struct.input_directory,
            output_directory=remote_dir_struct.output_directory,
        )
        logger.debug(f'Created job specification: {job_specification}')

        inbound_transfer_inputs = GlobusTransferInputs(
            source_url=create_globus_url(
                rc_id, rc_globus_path, *local_dir_struct.output_path_segments
            ),
            destination_url=create_globus_url(
                lc_id, lc_globus_path, *local_dir_struct.output_path_segments
            ),
            label='Transfer Inbound',
        )
        logger.debug(f'Created inbound transfer inputs: {inbound_transfer_inputs}')

        status_interval_s = self._settings.status_refresh_interval_s.get_value()
        load_product_path = (
            local_dir_struct.output_directory / StandardFileLayout.PRODUCT_OUT
            if ptychodus_action == 'reconstruct'
            else None
        )

        workflow_task = PtychodusWorkflow(
            product_api=self._product_api,
            transfer_client=transfer_client,
            compute_client=iri_client.compute,
            outbound_transfer_inputs=outbound_transfer_inputs,
            compute_resource_id=compute_resource_id,
            job_specification=job_specification,
            inbound_transfer_inputs=inbound_transfer_inputs,
            stop_event=self._stop_event,
            status_q=self._status_q,
            status_interval_s=status_interval_s,
            facility=facility_name,
            flow_label=flow_label,
            load_product_path=load_product_path,
        )
        self._task_manager.put_background_task(workflow_task)

    def reconstruct(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        dir_structure = self.populate_input_directory(input_product_index)

        if self._processing_api.is_reconstructor_trainable():
            ext = self._processing_api.get_model_file_extension()
            model_path = dir_structure.input_directory / f'{StandardFileLayout.MODEL_BASENAME}{ext}'
            self._processing_api.save_model_to_file(model_path)

        self._run_flow('reconstruct', dir_structure)

    def train(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        dir_structure = self.populate_input_directory(input_product_index)
        self._run_flow('train', dir_structure)
        # TODO customize input/output directories; put model to mlflow
