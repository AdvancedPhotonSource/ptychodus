from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class AgentSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Agent')
        self._group.add_observer(self)

        self.base_url = self._group.create_string_parameter(
            'BaseURL', 'https://apps.inside.anl.gov/argoapi/v1'
        )
        self.model = self._group.create_string_parameter('Model', 'GPT-4o')
        self.system_prompt = self._group.create_string_parameter(
            'SystemPrompt', 'You are a helpful assistant.'
        )
        self.temperature = self._group.create_real_parameter(
            'Temperature', 0.1, minimum=0.0, maximum=2.0
        )
        self.top_p = self._group.create_real_parameter('TopP', 0.9, minimum=0.0, maximum=1.0)
        self.max_tokens = self._group.create_integer_parameter(
            'MaxTokens', 1000, minimum=0, maximum=128000
        )
        self.mcp_server_url = self._group.create_string_parameter('MCPServerURL', '')
        self.database_path = self._group.create_string_parameter('DatabasePath', '')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
