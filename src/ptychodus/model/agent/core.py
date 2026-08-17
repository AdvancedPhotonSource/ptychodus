from ptychodus.api.settings import SettingsRegistry

from .model_catalog import ModelCatalog
from .repository import ConversationRepository
from .settings import AgentSettings
from .terminal import ChatTerminal


class AgentCore:
    def __init__(self, settings_registry: SettingsRegistry) -> None:
        self.settings = AgentSettings(settings_registry)
        self.repository = ConversationRepository(self.settings.database_path.get_value())
        self.terminal = ChatTerminal(self.settings, self.repository)
        self.catalog = ModelCatalog(self.settings)
