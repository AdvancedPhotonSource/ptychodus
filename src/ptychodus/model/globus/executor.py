from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import logging
import queue

from ptychodus.api.io import StandardFileLayout
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..product import ProductAPI
from .settings import GlobusSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobusJob:
    flow_label: str
    flow_input: Mapping[str, Any]


class GlobusExecutor:
    def __init__(
        self,
        settings: GlobusSettings,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._settings_registry = settings_registry
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self.job_queue: queue.Queue[GlobusJob] = queue.Queue()  # FIXME replace

    def run_flow(self, input_product_index: int) -> None:
        transfer_sync_level = 3  # Copy files if checksums of the source and destination mismatch
        ptychodus_action = 'reconstruct'  # TODO or 'train'

        try:
            flow_label = self._product_api.get_item(input_product_index).get_name()
        except IndexError:
            logger.warning(f'Failed access product for flow ({input_product_index=})!')
            return

        input_data_posix_path = self._settings.input_data_posix_path.get_value() / flow_label
        compute_data_posix_path = self._settings.compute_data_posix_path.get_value() / flow_label

        input_data_globus_path = f'{self._settings.input_data_globus_path.get_value()}/{flow_label}'
        compute_data_globus_path = (
            f'{self._settings.compute_data_globus_path.get_value()}/{flow_label}'
        )
        output_data_globus_path = (
            f'{self._settings.output_data_globus_path.get_value()}/{flow_label}'
        )

        try:
            input_data_posix_path.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.warning('Input data POSIX path must be a directory!')
            return

        self._settings_registry.save_settings(input_data_posix_path / StandardFileLayout.SETTINGS)
        self._diffraction_api.export_assembled_patterns(
            input_data_posix_path / StandardFileLayout.DIFFRACTION
        )
        self._product_api.save_product(
            input_product_index,
            input_data_posix_path / StandardFileLayout.PRODUCT_IN,
            file_type='HDF5',
        )

        flow_input = {
            'input_data_transfer_source_endpoint': str(
                self._settings.input_data_endpoint_id.get_value()
            ),
            'input_data_transfer_source_path': input_data_globus_path,
            'input_data_transfer_destination_endpoint': str(
                self._settings.compute_data_endpoint_id.get_value()
            ),
            'input_data_transfer_destination_path': compute_data_globus_path,
            'input_data_transfer_recursive': True,
            'input_data_transfer_sync_level': transfer_sync_level,
            'compute_endpoint': str(self._settings.compute_endpoint_id.get_value()),
            'ptychodus_action': ptychodus_action,
            'ptychodus_settings_file': str(compute_data_posix_path / StandardFileLayout.SETTINGS),
            'ptychodus_diffraction_file': str(
                compute_data_posix_path / StandardFileLayout.DIFFRACTION
            ),
            'ptychodus_input_file': str(compute_data_posix_path / StandardFileLayout.PRODUCT_IN),
            'ptychodus_output_file': str(compute_data_posix_path / StandardFileLayout.PRODUCT_OUT),
            'output_data_transfer_source_endpoint': str(
                self._settings.compute_data_endpoint_id.get_value()
            ),
            'output_data_transfer_source_path': f'{compute_data_globus_path}/{StandardFileLayout.PRODUCT_OUT}',
            'output_data_transfer_destination_endpoint': str(
                self._settings.output_data_endpoint_id.get_value()
            ),
            'output_data_transfer_destination_path': f'{output_data_globus_path}/{StandardFileLayout.PRODUCT_OUT}',
            'output_data_transfer_recursive': False,
        }

        input_ = GlobusJob(flow_label, flow_input)
        self.job_queue.put(input_)
