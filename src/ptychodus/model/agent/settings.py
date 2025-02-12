import getpass

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ArgoSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.createGroup('Argo')
        self._group.addObserver(self)

        self.user = self._group.createStringParameter('User', getpass.getuser())
        self.chatEndpointURL = self._group.createStringParameter(
            'ChatEndpointURL', 'https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/'
        )
        self.chatModel = self._group.createStringParameter('ChatModel', 'gpt35')
        self.temperature = self._group.createRealParameter(
            'Temperature', 0.1, minimum=0.0, maximum=2.0
        )
        self.top_p = self._group.createRealParameter('TopP', 0.9, minimum=0.0, maximum=1.0)
        self.max_tokens = self._group.createIntegerParameter(
            'MaxTokens', 1000, minimum=0, maximum=128000
        )
        self.max_completion_tokens = self._group.createIntegerParameter(
            'MaxCompletionTokens', 1000, minimum=0, maximum=128000
        )
        self.embeddingsEndpointURL = self._group.createStringParameter(
            'EmbeddingsEndpointURL', 'https://apps.inside.anl.gov/argoapi/api/v1/resource/embed/'
        )
        self.embeddingsModel = self._group.createStringParameter('EmbeddingsModel', 'ada002')

    def update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notifyObservers()
