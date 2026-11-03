from ptychodus.api.language_model import LanguageModelProvider

from ..plugins import PluginChooser
from ..settings import SettingsRegistry
from .model_catalog import ModelCatalog
from .provider import LanguageModelProviderSelection
from .repository import ConversationRepository
from .settings import AgentSettings
from .terminal import ChatTerminal


class AgentCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        provider_chooser: PluginChooser[LanguageModelProvider],
    ) -> None:
        self.settings = AgentSettings(settings_registry)
        self.provider_selection = LanguageModelProviderSelection(
            settings_registry, self.settings, provider_chooser
        )
        self.repository = ConversationRepository(self.settings.database_path.get_value())
        self.terminal = ChatTerminal(self.settings, self.provider_selection, self.repository)
        self.catalog = ModelCatalog(self.provider_selection)
