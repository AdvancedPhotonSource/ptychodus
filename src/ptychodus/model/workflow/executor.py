from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import logging
import queue

from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..product import ProductAPI
from .locator import DataLocator
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowJob:
    flow_label: str
    flow_input: Mapping[str, Any]


class WorkflowExecutor:
    def __init__(
        self,
        settings: WorkflowSettings,
        input_data_locator: DataLocator,
        compute_data_locator: DataLocator,
        output_data_locator: DataLocator,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._input_data_locator = input_data_locator
        self._compute_data_locator = compute_data_locator
        self._output_data_locator = output_data_locator
        self._product_api = product_api
        self._settings_registry = settings_registry
        self._diffraction_api = diffraction_api
        self.job_queue: queue.Queue[WorkflowJob] = queue.Queue()

    def run_flow(self, input_product_index: int) -> None:
        transfer_sync_level = 3  # Copy files if checksums of the source and destination mismatch
        ptychodus_action = 'reconstruct'  # TODO or 'train'

        try:
            flow_label = self._product_api.get_item(input_product_index).get_name()
        except IndexError:
            logger.warning(f'Failed access product for flow ({input_product_index=})!')
            return

        input_data_posix_path = self._input_data_locator.get_posix_path() / flow_label
        compute_data_posix_path = self._compute_data_locator.get_posix_path() / flow_label

        input_data_globus_path = f'{self._input_data_locator.get_globus_path()}/{flow_label}'
        compute_data_globus_path = f'{self._compute_data_locator.get_globus_path()}/{flow_label}'
        output_data_globus_path = f'{self._output_data_locator.get_globus_path()}/{flow_label}'

        settings_file = 'settings.ini'
        patterns_file = 'patterns.h5'
        input_file = 'product-in.h5'
        output_file = 'product-out.h5'

        try:
            input_data_posix_path.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.warning('Input data POSIX path must be a directory!')
            return

        # TODO use workflow API
        self._settings_registry.save_settings(input_data_posix_path / settings_file)
        self._diffraction_api.export_assembled_patterns(input_data_posix_path / patterns_file)
        self._product_api.save_product(
            input_product_index, input_data_posix_path / input_file, file_type='HDF5'
        )

        flow_input = {
            'input_data_transfer_source_endpoint': str(self._input_data_locator.get_endpoint_id()),
            'input_data_transfer_source_path': input_data_globus_path,
            'input_data_transfer_destination_endpoint': str(
                self._compute_data_locator.get_endpoint_id()
            ),
            'input_data_transfer_destination_path': compute_data_globus_path,
            'input_data_transfer_recursive': True,
            'input_data_transfer_sync_level': transfer_sync_level,
            'compute_endpoint': str(self._settings.compute_endpoint_id.get_value()),
            'ptychodus_action': ptychodus_action,
            'ptychodus_settings_file': str(compute_data_posix_path / settings_file),
            'ptychodus_patterns_file': str(compute_data_posix_path / patterns_file),
            'ptychodus_input_file': str(compute_data_posix_path / input_file),
            'ptychodus_output_file': str(compute_data_posix_path / output_file),
            'output_data_transfer_source_endpoint': str(
                self._compute_data_locator.get_endpoint_id()
            ),
            'output_data_transfer_source_path': f'{compute_data_globus_path}/{output_file}',
            'output_data_transfer_destination_endpoint': str(
                self._output_data_locator.get_endpoint_id()
            ),
            'output_data_transfer_destination_path': f'{output_data_globus_path}/{output_file}',
            'output_data_transfer_recursive': False,
        }

        input_ = WorkflowJob(flow_label, flow_input)
        self.job_queue.put(input_)
