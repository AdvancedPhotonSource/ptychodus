from collections.abc import Iterator, Sequence
import logging

from ptychodus.api.settings import SettingsRegistry

from .chat import ChatHistory, ChatTerminal
from .settings import ArgoSettings

logger = logging.getLogger(__name__)


class AgentPresenter:
    def __init__(self, terminal: ChatTerminal | None) -> None:
        self._terminal = terminal

    @property
    def is_agent_supported(self) -> bool:
        return self._terminal is not None

    def get_available_chat_models(self) -> Iterator[str]:
        for model in [
            'gpt35',
            'gpt35large',
            'gpt4',
            'gpt4large',
            'gpt4o',
            'gpt4turbo',
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
    # FIXME langchain-chroma for RAG

    def __init__(self, settings_registry: SettingsRegistry):
        self.settings = ArgoSettings(settings_registry)
        self.chat_history = ChatHistory()
        self._terminal: ChatTerminal | None = None

        try:
            from .argo import ArgoChatTerminal
        except ModuleNotFoundError:
            logger.info('langchain not found!')
        else:
            self._terminal = ArgoChatTerminal(self.settings, self.chat_history)

        self.presenter = AgentPresenter(self._terminal)

    @property
    def is_supported(self) -> bool:
        return self.presenter.is_agent_supported
