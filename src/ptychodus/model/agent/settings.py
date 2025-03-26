import getpass

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ArgoSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Argo')
        self._group.add_observer(self)

        self.user = self._group.create_string_parameter('User', getpass.getuser())
        self.chat_endpoint_url = self._group.create_string_parameter(
            'ChatEndpointURL', 'https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/'
        )
        self.chat_model = self._group.create_string_parameter('ChatModel', 'gpt35')
        self.temperature = self._group.create_real_parameter(
            'Temperature', 0.1, minimum=0.0, maximum=2.0
        )
        self.top_p = self._group.create_real_parameter('TopP', 0.9, minimum=0.0, maximum=1.0)
        self.max_tokens = self._group.create_integer_parameter(
            'MaxTokens', 1000, minimum=0, maximum=128000
        )
        self.max_completion_tokens = self._group.create_integer_parameter(
            'MaxCompletionTokens', 1000, minimum=0, maximum=128000
        )
        self.embeddings_endpoint_url = self._group.create_string_parameter(
            'EmbeddingsEndpointURL', 'https://apps.inside.anl.gov/argoapi/api/v1/resource/embed/'
        )
        self.embeddings_model = self._group.create_string_parameter('EmbeddingsModel', 'ada002')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
