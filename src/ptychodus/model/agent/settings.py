from enum import StrEnum

from ..observer import Observable, Observer
from ..settings import SettingsRegistry


class ThinkingLevel(StrEnum):
    DEFAULT = 'Default'
    OFF = 'Off'
    LOW = 'Low'
    MEDIUM = 'Medium'
    HIGH = 'High'


class AgentSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Agent')
        self._group.add_observer(self)

        self.base_url = self._group.create_string_parameter(
            'BaseURL', 'https://apps.inside.anl.gov/argoapi'
        )
        self.model = self._group.create_string_parameter('Model', 'claudesonnet55')
        self.instructions = self._group.create_string_parameter(
            'Instructions', 'You are a helpful assistant.'
        )
        self.thinking = self._group.create_string_parameter('Thinking', ThinkingLevel.DEFAULT)
        self.max_tokens = self._group.create_integer_parameter(
            'MaxTokens', 0, minimum=0, maximum=128000
        )
        self.mcp_server_url = self._group.create_string_parameter('MCPServerURL', '')
        self.database_path = self._group.create_string_parameter('DatabasePath', '')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
