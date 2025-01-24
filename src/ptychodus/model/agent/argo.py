from collections.abc import Sequence
import json
import logging
import requests

from .settings import ArgoSettings

logger = logging.getLogger(__name__)


class Argo:
    def __init__(self, settings: ArgoSettings):
        self._settings = settings

    def invoke(self, prompt: Sequence[str], system: str, stop: Sequence[str] = []) -> str:
        # Data to be sent as a POST in JSON format
        data = {
            'user': self._settings.user.getValue(),
            'model': self._settings.model.getValue(),
            'system': system,
            'prompt': prompt,
            'stop': stop,
            'temperature': self._settings.temperature.getValue(),
            'top_p': self._settings.top_p.getValue(),
            'max_tokens': self._settings.max_tokens.getValue(),
            'max_completion_tokens': self._settings.max_completion_tokens.getValue(),
        }

        # Send POST request
        payload = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            self._settings.chatEndpointURL.getValue(), data=payload, headers=headers
        )

        # Receive the response data
        logger.debug('Status Code:', response.status_code)
        response_json = response.json()
        logger.debug('JSON Response:', response_json)

        return response_json['response']
