import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from .executor import GenesisExecutor
from .iri import IRIClient, get_iri_tokens_file
from .settings import GenesisSettings
from .transfer import GenesisGlobusTransferClient

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI
from ..task_manager import TaskManager
from .tokens import get_transfer_tokens_file, read_tokens

__all__ = [
    'GenesisCore',
]

logger = logging.getLogger(__name__)


def create_iri_client_chooser() -> PluginChooser[IRIClient]:
    plugin_chooser = PluginChooser[IRIClient]()
    tokens_file = get_iri_tokens_file()
    logger.info(f'Loading IRI access tokens from {tokens_file}...')

    if tokens_file.is_file():
        for tokens in read_tokens(tokens_file):
            plugin_chooser.register_plugin(
                IRIClient(api_base_url=tokens.api_base_url, access_token=tokens.access_token),
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
        self.iri_client_chooser = create_iri_client_chooser()
        self.iri_client_chooser.synchronize_with_parameter(self.settings.iri_provider)
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
            self.iri_client_chooser,
            self.transfer_client_chooser,
        )

    @property
    def is_supported(self) -> bool:
        return bool(self.iri_client_chooser) and bool(self.transfer_client_chooser)
