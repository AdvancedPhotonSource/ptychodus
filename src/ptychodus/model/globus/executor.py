from pathlib import Path
import logging

from ptychodus.api.io import sanitize_path_component

from ..product import ProductAPI
from ..processing import ProcessingAPI
from .client import GlobusClient, GlobusJob
from .settings import GlobusSettings

logger = logging.getLogger(__name__)


class GlobusExecutor:
    def __init__(
        self,
        settings: GlobusSettings,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
        client: GlobusClient,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._product_api = product_api
        self._processing_api = processing_api
        self._client = client

    def populate_input_directory(self, input_product_index: int, action: str) -> Path:
        """Stage `action`'s inputs for the product under ``<name>/input/``; return ``<name>``."""
        try:
            product_item = self._product_api.get_item(input_product_index)
        except IndexError:
            logger.exception(f'Failed access product for flow ({input_product_index=})!')
            raise

        base_directory = self._settings.input_collection_posix_path.get_value() / (
            sanitize_path_component(product_item.get_name())
        )
        input_directory = base_directory / 'input'

        try:
            input_directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.exception('Input data POSIX path must be a directory!')
            raise

        if action == 'train':
            self._processing_api.stage_train_inputs(
                input_directory, product_index=input_product_index
            )
        else:
            self._processing_api.stage_reconstruct_inputs(
                input_directory, product_index=input_product_index
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

        # Remote-workflow directory layout mirrors Genesis: input files sit
        # under <flow_label>/input/, output files are written to <flow_label>/output/.
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

    def reconstruct(
        self,
        input_product_index: int,
        *,
        library: str | None = None,
        algorithm: str | None = None,
    ) -> None:
        self._processing_api.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        base_directory = self.populate_input_directory(input_product_index, 'reconstruct')
        self._run_flow('reconstruct', base_directory.name)

    def train(
        self,
        input_product_index: int,
        *,
        library: str | None = None,
        algorithm: str | None = None,
    ) -> None:
        self._processing_api.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        base_directory = self.populate_input_directory(input_product_index, 'train')
        self._run_flow('train', base_directory.name)
