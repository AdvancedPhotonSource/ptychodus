from collections.abc import Iterator
import logging

from ptychodus.api.settings import SettingsRegistry

from .argo import Argo
from .repository import ChatMessage, ChatMessageSender, ChatRepository
from .settings import ArgoSettings

logger = logging.getLogger(__name__)


class AgentPresenter:
    def __init__(self, repository: ChatRepository, argo: Argo) -> None:
        self._repository = repository
        self._argo = argo

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

    def send_message(self, text: str) -> None:
        if text:
            # FIXME system = 'You are a large language model with the name Argo.'
            prompt = text.splitlines()
            # FIXME self._argo.invoke(prompt, system)
            logger.info(prompt)

            message = ChatMessage(sender=ChatMessageSender.HUMAN, contents=text)
            self._repository.append(message)


class AgentCore:
    def __init__(self, settingsRegistry: SettingsRegistry):
        self.argoSettings = ArgoSettings(settingsRegistry)
        self.chatRepository = ChatRepository()
        self._argo = Argo(self.argoSettings)
        self.presenter = AgentPresenter(self.chatRepository, self._argo)
