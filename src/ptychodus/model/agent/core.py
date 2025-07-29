from collections.abc import Iterator, Sequence
import logging

from ptychodus.api.settings import SettingsRegistry

from .argo import ArgoChatTerminal
from .chat import ChatHistory, ChatTerminal
from .settings import ArgoSettings

logger = logging.getLogger(__name__)


class AgentPresenter:
    def __init__(self, terminal: ChatTerminal) -> None:
        self._terminal = terminal

    def get_available_chat_models(self) -> Iterator[str]:
        for model in [
            'gpt35',
            'gpt35large',
            'gpt4',
            'gpt4large',
            'gpt4o',
            'gpt4olatestgpt4turbo',
            'gpto1',
            'gpto1mini',
            'gpto3mini',
        ]:
            yield model

    def send_message(self, content: str) -> None:
        if self._terminal is not None:
            self._terminal.send_message(content)

    def get_available_embeddings_models(self) -> Iterator[str]:
        for model in ['ada002', 'v3large', 'v3small']:
            yield model

    def embed_text(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[]] if self._terminal is None else self._terminal.embed_texts(texts)


class AgentCore:
    def __init__(self, settings_registry: SettingsRegistry):
        self.settings = ArgoSettings(settings_registry)
        self.chat_history = ChatHistory()
        self._terminal = ArgoChatTerminal(self.settings, self.chat_history)
        self.presenter = AgentPresenter(self._terminal)
