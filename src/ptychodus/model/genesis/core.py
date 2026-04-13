import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from .compute import GenesisComputeClient
from .settings import GenesisSettings
from .transfer import GenesisGlobusTransferClient
from .executor import GenesisExecutor

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI
from ..task_manager import TaskManager
from .tokens import (
    get_compute_tokens_file,
    get_transfer_tokens_file,
    read_tokens,
)

__all__ = [
    'GenesisCore',
]

logger = logging.getLogger(__name__)


def create_compute_client_chooser() -> PluginChooser[GenesisComputeClient]:
    plugin_chooser = PluginChooser[GenesisComputeClient]()
    tokens_file = get_compute_tokens_file()
    logger.info(f'Loading compute access tokens from {tokens_file}...')

    if tokens_file.is_file():
        for tokens in read_tokens(tokens_file):
            plugin_chooser.register_plugin(
                GenesisComputeClient(
                    api_base_url=tokens.api_base_url, access_token=tokens.access_token
                ),
                display_name=tokens.name,
            )
    else:
        logger.debug(f'Compute access tokens file {tokens_file} does not exist.')

    return plugin_chooser


def create_transfer_client_chooser() -> PluginChooser[GenesisGlobusTransferClient]:
    plugin_chooser = PluginChooser[GenesisGlobusTransferClient]()
    tokens_file = get_transfer_tokens_file()
    logger.info(f'Loading transfer access tokens from {tokens_file}...')

    if tokens_file.is_file():
        for tokens in read_tokens(tokens_file):
            plugin_chooser.register_plugin(
                GenesisGlobusTransferClient(
                    api_base_url=tokens.api_base_url, access_token=tokens.access_token
                ),
                display_name=tokens.name,
            )
    else:
        logger.debug(f'Transfer access tokens file {tokens_file} does not exist.')

    return plugin_chooser


class GenesisCore:
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
    ) -> None:
        self.settings = GenesisSettings(settings_registry)
        self.compute_client_chooser = create_compute_client_chooser()
        self.compute_client_chooser.synchronize_with_parameter(self.settings.compute_provider)
        self.transfer_client_chooser = create_transfer_client_chooser()
        self.transfer_client_chooser.synchronize_with_parameter(
            self.settings.globus_transfer_provider
        )
        self.executor = GenesisExecutor(
            task_manager,
            settings_registry,
            diffraction_api,
            product_api,
            processing_api,
            self.settings,
            self.compute_client_chooser,
            self.transfer_client_chooser,
        )

    @property
    def is_supported(self) -> bool:
        return bool(self.compute_client_chooser) and bool(self.transfer_client_chooser)
