from pathlib import Path
import logging

from ptychodus.api.io import StandardFileLayout, sanitize_path_component
from ptychodus.api.settings import SettingsRegistry

from ..product import ProductAPI
from ..processing import ProcessingAPI
from .client import GlobusClient, GlobusJob
from .settings import GlobusSettings

logger = logging.getLogger(__name__)


class GlobusExecutor:
    def __init__(
        self,
        settings: GlobusSettings,
        settings_registry: SettingsRegistry,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
        client: GlobusClient,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._settings_registry = settings_registry
        self._product_api = product_api
        self._processing_api = processing_api
        self._client = client

    def populate_input_directory(self, input_product_index: int) -> Path:
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

        base_directory = self._settings.input_collection_posix_path.get_value() / (
            sanitize_path_component(product_item.get_name())
        )
        input_directory = base_directory / 'input'

        try:
            input_directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.exception('Input data POSIX path must be a directory!')
            raise

        self._settings_registry.save_settings(input_directory / StandardFileLayout.SETTINGS)
        dataset.export_assembled_patterns(input_directory / StandardFileLayout.DIFFRACTION)
        self._product_api.save_product(
            input_product_index,
            input_directory / StandardFileLayout.PRODUCT,
            file_type='HDF5',
        )

        return base_directory

    def _run_flow(self, ptychodus_action: str, flow_label: str) -> None:
        compute_data_posix_path = (
            self._settings.compute_collection_posix_path.get_value() / flow_label
        )
        compute_input_posix_path = compute_data_posix_path / 'input'
        compute_output_posix_path = compute_data_posix_path / 'output'

        input_collection_globus_path = self._settings.input_collection_globus_path.get_value()
        compute_collection_globus_path = self._settings.compute_collection_globus_path.get_value()
        output_collection_globus_path = self._settings.output_collection_globus_path.get_value()

        # Remote-workflow directory layout mirrors Genesis: input artifacts sit
        # under <flow_label>/input/, results are written to <flow_label>/output/.
        # Transfer only the relevant subtree in each direction.
        input_data_globus_path = f'{input_collection_globus_path}/{flow_label}/input'
        compute_input_globus_path = f'{compute_collection_globus_path}/{flow_label}/input'
        compute_output_globus_path = f'{compute_collection_globus_path}/{flow_label}/output'
        output_data_globus_path = f'{output_collection_globus_path}/{flow_label}/output'

        flow_input = {
            'transfer_input_data': {
                'source': {
                    'id': str(self._settings.input_collection_id.get_value()),
                    'path': input_data_globus_path,
                },
                'destination': {
                    'id': str(self._settings.compute_collection_id.get_value()),
                    'path': compute_input_globus_path,
                },
                'sync_level': self._settings.transfer_sync_level.get_value(),
                'recursive': True,
            },
            'compute': {
                'endpoint_id': str(self._settings.compute_endpoint_id.get_value()),
                # NOTE: 'function_id': compute_function_id, # added in globus.py
                'function_kwargs': {
                    'action': ptychodus_action,
                    'input_directory': str(compute_input_posix_path),
                    'output_directory': str(compute_output_posix_path),
                },
            },
            'transfer_output_data': {
                'source': {
                    'id': str(self._settings.compute_collection_id.get_value()),
                    'path': compute_output_globus_path,
                },
                'destination': {
                    'id': str(self._settings.output_collection_id.get_value()),
                    'path': output_data_globus_path,
                },
                'sync_level': self._settings.transfer_sync_level.get_value(),
                'recursive': True,
            },
        }

        flow_tags = ['aps', 'ptychography']
        job = GlobusJob(flow_input, flow_label, flow_tags)
        self._client.run_flow(job)

    def reconstruct(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        base_directory = self.populate_input_directory(input_product_index)

        if self._processing_api.is_reconstructor_trainable():
            pass  # TODO get model from mlflow

        self._run_flow('reconstruct', base_directory.name)

    def train(self, input_product_index: int, *, algorithm: str | None = None) -> None:
        self._processing_api.set_reconstructor_if_provided(algorithm)
        base_directory = self.populate_input_directory(input_product_index)
        self._run_flow('train', base_directory.name)
        # TODO customize input/output directories; put model to mlflow
