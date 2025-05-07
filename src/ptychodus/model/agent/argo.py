from collections.abc import Sequence
import logging
import requests

from .chat import ChatHistory, ChatMessage, ChatRole, ChatTerminal
from .settings import ArgoSettings

logger = logging.getLogger(__name__)


class ArgoChatTerminal(ChatTerminal):
    def __init__(self, settings: ArgoSettings, history: ChatHistory) -> None:
        self._settings = settings
        self._history = history

    def send_message(self, content: str, stop: Sequence[str] = []) -> None:
        if not content:
            return

        messages = [
            ChatMessage(
                role=ChatRole.SYSTEM, content='You are a large language model with the name Argo.'
            )
        ]

        for line in content.splitlines():
            message = ChatMessage(role=ChatRole.USER, content=line)
            messages.append(message)
            self._history.add_message(message)

        logger.debug(f'{messages=}')

        url = self._settings.chat_endpoint_url.get_value()
        payload = {
            'user': self._settings.user.get_value(),
            'model': self._settings.chat_model.get_value(),
            'messages': [m.to_dict() for m in messages],
            'stop': stop,
            'temperature': self._settings.temperature.get_value(),
            'top_p': self._settings.top_p.get_value(),
            'max_tokens': self._settings.max_tokens.get_value(),
            'max_completion_tokens': self._settings.max_completion_tokens.get_value(),
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)

        logger.debug(f'{response=}')
        logger.debug(f'Status Code: {response.status_code}')
        response_json = response.json()
        logger.debug(f'JSON Response: {response_json}')
        response.raise_for_status()

        response_message = ChatMessage(role=ChatRole.AGENT, content=response_json['response'])
        self._history.add_message(response_message)

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Generates embeddings for a list of strings."""
        url = self._settings.embeddings_endpoint_url.get_value()
        payload = {
            'user': self._settings.user.get_value(),
            'model': self._settings.embeddings_model.get_value(),
            'prompt': texts,
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=payload, headers=headers)

        logger.debug(response)
        logger.debug(f'Status Code: {response.status_code}')
        response_json = response.json()
        logger.debug(f'JSON Response: {response_json}')
        response.raise_for_status()

        return response_json['embedding']
