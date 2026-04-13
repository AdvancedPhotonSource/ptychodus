from pathlib import Path
import logging

from ptychodus.api.io import StandardFileLayout
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI
from ..task_manager import BackgroundTaskManager, ForegroundTask
from .compute import GenesisComputeClient
from .settings import GenesisSettings
from .transfer import GenesisGlobusTransferClient

logger = logging.getLogger(__name__)


class WorkflowTask:
    def __init__(
        self,
        transfer_client: GenesisGlobusTransferClient,
        compute_client: GenesisComputeClient,
        ptychodus_action: str,
        flow_label: str,
    ) -> None:
        super().__init__()
        self._transfer_client = transfer_client
        self._compute_client = compute_client
        self._ptychodus_action = ptychodus_action
        self._flow_label = flow_label

    def __call__(self) -> ForegroundTask | None:
        logger.info(f'Executing workflow task ({self._ptychodus_action=}, {self._flow_label=})...')
        # TODO status updates
        # TODO transfer inputs
        # TODO submit compute job
        # TODO transfer outputs
        # TODO load result
        return None  # FIXME


class GenesisExecutor:
    def __init__(
        self,
        task_manager: BackgroundTaskManager,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
        settings: GenesisSettings,
        compute_client_chooser: PluginChooser[GenesisComputeClient],
        transfer_client_chooser: PluginChooser[GenesisGlobusTransferClient],
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._settings = settings
        self._settings_registry = settings_registry
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self._processing_api = processing_api
        self._compute_client_chooser = compute_client_chooser
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

    def _run_flow(self, ptychodus_action: str, flow_label: str) -> None:
        workflow_task = WorkflowTask(
            self._transfer_client_chooser.get_current_plugin().strategy,
            self._compute_client_chooser.get_current_plugin().strategy,
            ptychodus_action,
            flow_label,
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
