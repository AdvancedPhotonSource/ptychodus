from collections.abc import Mapping, Sequence
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI
from ..task_manager import TaskManager
from .executor import GenesisExecutor
from .facility_adapters import IRIFacilityAdapter, ALCFFacilityAdapter, NERSCFacilityAdapter
from .iri import get_iri_tokens_file
from .settings import GenesisSettings
from .tokens import read_tokens
from .transfer import AmSCGlobusTransferClient, get_amsc_transfer_api_url, get_transfer_tokens_file

__all__ = [
    'GenesisCore',
    'GenesisPresenter',
    'create_facility_adapters',
    'create_globus_transfer_providers',
]

logger = logging.getLogger(__name__)


def create_facility_adapters() -> Mapping[str, IRIFacilityAdapter]:
    adapters: dict[str, IRIFacilityAdapter] = {}

    tokens_file = get_iri_tokens_file()
    logger.info(f'Loading IRI access tokens from {tokens_file}...')
    access_tokens = read_tokens(tokens_file)

    for token in access_tokens:
        facility = token.facility.casefold()
        if facility == ALCFFacilityAdapter.NAME.casefold():
            adapters[ALCFFacilityAdapter.NAME] = ALCFFacilityAdapter(token.access_token)
        elif facility == NERSCFacilityAdapter.NAME.casefold():
            adapters[NERSCFacilityAdapter.NAME] = NERSCFacilityAdapter(token.access_token)
        else:
            logger.warning(f'Unsupported facility: {token.facility}')

    return adapters


def create_globus_transfer_providers() -> Mapping[str, AmSCGlobusTransferClient]:
    providers: dict[str, AmSCGlobusTransferClient] = {}

    tokens_file = get_transfer_tokens_file()
    logger.info(f'Loading transfer access tokens from {tokens_file}...')
    access_tokens = read_tokens(tokens_file)

    for token in access_tokens:
        facility = token.facility.casefold()
        if facility == AmSCGlobusTransferClient.NAME.casefold():
            providers[AmSCGlobusTransferClient.NAME] = AmSCGlobusTransferClient(
                api_base_url=get_amsc_transfer_api_url(), access_token=token.access_token
            )
        else:
            logger.warning(f'Unsupported transfer provider: {token.facility}')

    return providers


class GenesisPresenter:
    def __init__(
        self,
        settings: GenesisSettings,
        facility_chooser: PluginChooser[IRIFacilityAdapter],
        transfer_client_chooser: PluginChooser[AmSCGlobusTransferClient],
    ) -> None:
        self._settings = settings
        self._facility_chooser = facility_chooser
        self._transfer_client_chooser = transfer_client_chooser

    def supported_facilities(self) -> Sequence[str]:
        return [plugin.display_name for plugin in self._facility_chooser]

    def supported_transfer_clients(self) -> Sequence[str]:
        return [plugin.display_name for plugin in self._transfer_client_chooser]

    def apply_facility_defaults(self, label: str) -> None:
        self._facility_chooser.set_current_plugin(label)  # FIXME check this
        adapter = self._facility_chooser.get_current_plugin().strategy

        compute_resource_id = adapter.get_default_compute_resource_id()
        self._settings.compute_resource_id.set_value(str(compute_resource_id))

        globus_collection = adapter.get_default_globus_collection()
        self._settings.remote_collection_id.set_value(globus_collection.id)
        self._settings.remote_collection_globus_path.set_value(globus_collection.globus_path)
        self._settings.remote_collection_posix_path.set_value(globus_collection.posix_path)


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

        self._facility_chooser = PluginChooser[IRIFacilityAdapter]()
        self._transfer_client_chooser = PluginChooser[AmSCGlobusTransferClient]()

        for name, adapter in create_facility_adapters().items():
            self._facility_chooser.register_plugin(adapter, display_name=name)

        for name, provider in create_globus_transfer_providers().items():
            self._transfer_client_chooser.register_plugin(provider, display_name=name)

        self._facility_chooser.synchronize_with_parameter(self.settings.facility)
        self._transfer_client_chooser.synchronize_with_parameter(
            self.settings.globus_transfer_provider
        )

        self.executor = GenesisExecutor(
            task_manager,
            settings_registry,
            diffraction_api,
            product_api,
            processing_api,
            self.settings,
            self._facility_chooser,
            self._transfer_client_chooser,
        )
        self.presenter = GenesisPresenter(
            self.settings, self._facility_chooser, self._transfer_client_chooser
        )

    @property
    def is_supported(self) -> bool:
        return bool(self._facility_chooser) and bool(self._transfer_client_chooser)
