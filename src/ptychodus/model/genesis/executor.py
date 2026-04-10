from pathlib import Path
import logging

from ptychodus.api.io import StandardFileLayout
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI
from .settings import GenesisSettings

logger = logging.getLogger(__name__)


class GenesisExecutor:
    def __init__(
        self,
        settings: GenesisSettings,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._settings_registry = settings_registry
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self._processing_api = processing_api

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
        pass  # FIXME

    def reconstruct(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        input_directory = self.populate_input_directory(input_product_index)
        self._run_flow('reconstruct', input_directory.name)

    def train(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        # TODO customize input_directory and output_directory
        self._processing_api.set_reconstructor_if_provided(algorithm)
        input_directory = self.populate_input_directory(input_product_index)
        self._run_flow('train', input_directory.name)  # TODO mlflow

    def infer(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        input_directory = self.populate_input_directory(input_product_index)
        self._run_flow('infer', input_directory.name)  # TODO mlflow
