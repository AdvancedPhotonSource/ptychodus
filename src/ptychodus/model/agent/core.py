from collections.abc import Iterator
import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ptychodus.api.settings import SettingsRegistry

from .argo import ChatArgo
from .chat import ChatHistory, ChatMessage, ChatRole
from .settings import ArgoSettings

logger = logging.getLogger(__name__)


class ChatFacade:
    def __init__(self, settings: ArgoSettings, history: ChatHistory) -> None:
        self._argo = ChatArgo(settings=settings)
        self._history = history

    def send_message(self, content: str) -> None:
        if content:
            system = SystemMessage(content='You are a large language model with the name Argo.')
            messages: list[BaseMessage] = [system]

            for line in content.splitlines():
                message = HumanMessage(content=line)
                messages.append(message)

                hmessage = ChatMessage(role=ChatRole.HUMAN, content=str(message.content))
                self._history.add_message(hmessage)

            logger.debug(f'{messages=}')
            response = self._argo.invoke(messages)
            logger.debug(f'{response=}')

            hmessage = ChatMessage(role=ChatRole.AI, content=str(response.content))
            self._history.add_message(hmessage)


class AgentPresenter:
    def __init__(self, facade: ChatFacade) -> None:
        self._facade = facade

    def get_available_models(self) -> Iterator[str]:
        for model in [
            'gpt35',
            'gpt35large',
            'gpt4',
            'gpt4large',
            'gpt4turbo',
            'gpt4o',
            'gpto1preview',
        ]:
            yield model

    def send_message(self, content: str) -> None:
        self._facade.send_message(content)


class AgentCore:
    # FIXME langchain-chroma for RAG

    def __init__(self, settings_registry: SettingsRegistry):
        self.settings = ArgoSettings(settings_registry)
        self.history = ChatHistory()
        self._facade = ChatFacade(self.settings, self.history)
        self.presenter = AgentPresenter(self._facade)
