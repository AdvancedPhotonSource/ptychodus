from pathlib import Path
from uuid import UUID
import logging
import os
import queue
import threading

from ptychodus.api.io import StandardFileLayout
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI
from ..task_manager import BackgroundTaskManager
from .facility_adapters import IRIFacilityAdapter
from .settings import GenesisSettings
from .transfer import AmSCGlobusTransferClient, GlobusTransferInputs
from .status import GenesisStatus
from .workflow import PtychodusWorkflow

logger = logging.getLogger(__name__)


def combine_path_segments(base_path: str, subpath: str) -> str:
    return f'{base_path.rstrip("/")}/{subpath.lstrip("/")}'


def create_globus_url(collection_id: UUID, globus_path: str) -> str:
    return f'globus://{collection_id}/{globus_path.lstrip("/")}'


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
        stop_event: threading.Event,
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
        self._stop_event = stop_event

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
        lc_posix_path = self._settings.local_collection_posix_path.get_value()

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
        logger.debug(f'Created outbound transfer inputs: {outbound_transfer_inputs}')
        job_specification = facility_adapter.create_job_specification(
            action=ptychodus_action,
            input_directory=Path(combine_path_segments(rc_posix_path, input_path_segment)),
            output_directory=Path(combine_path_segments(rc_posix_path, output_path_segment)),
        )
        logger.debug(f'Created job specification: {job_specification}')
        inbound_transfer_inputs = GlobusTransferInputs(
            source_url=create_globus_url(rc_id, inbound_source_path),
            destination_url=create_globus_url(lc_id, inbound_destination_path),
            label=f'ptychodus_{flow_label}_inbound',
            source_uuid=str(rc_id),
            source_path=inbound_source_path,
            destination_uuid=str(lc_id),
            destination_path=inbound_destination_path,
        )
        logger.debug(f'Created inbound transfer inputs: {inbound_transfer_inputs}')

        status_interval_s = self._settings.status_refresh_interval_s.get_value()
        load_product_path = (
            Path(os.path.join(lc_posix_path, output_path_segment)) / StandardFileLayout.PRODUCT_OUT
            if load_product
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
            label=flow_label,
            load_product_path=load_product_path,
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
