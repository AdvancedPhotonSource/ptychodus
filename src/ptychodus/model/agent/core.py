from collections.abc import Iterator
import logging

from ptychodus.api.settings import SettingsRegistry

from .argo import Argo
from .settings import ArgoSettings

logger = logging.getLogger(__name__)


class AgentPresenter:
    def __init__(self, argo: Argo) -> None:
        self._argo = argo
        self._messages: list[str] = []

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

    def get_message(self, index: int) -> str:
        return self._messages[index]

    def get_number_of_messages(self) -> int:
        return len(self._messages)

    def send_message(self, message: str) -> None:
        if message:
            # FIXME system = 'You are a large language model with the name Argo.'
            prompt = message.splitlines()
            # FIXME self._argo.invoke(prompt, system)
            logger.info(prompt)
            self._messages.append(message)


class AgentCore:
    def __init__(self, settingsRegistry: SettingsRegistry):
        self.argoSettings = ArgoSettings(settingsRegistry)
        self._argo = Argo(self.argoSettings)
        self.presenter = AgentPresenter(self._argo)
