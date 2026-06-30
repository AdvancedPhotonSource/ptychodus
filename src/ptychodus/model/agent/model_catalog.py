import logging
import os

import httpx

from .settings import AgentSettings

logger = logging.getLogger(__name__)


class ModelCatalog:
    """Fetches and caches the list of chat models advertised by the configured endpoint."""

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings
        self._cached_models: list[str] | None = None

    def get_available_models(self) -> list[str]:
        if self._cached_models is None:
            return self.refresh()
        return self._cached_models

    def refresh(self) -> list[str]:
        base_url = self._settings.base_url.get_value().rstrip('/')
        api_key = os.environ.get('OPENAI_API_KEY', '')
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        try:
            response = httpx.get(f'{base_url}/models', timeout=10.0, headers=headers)
            response.raise_for_status()
            data = response.json()['data']
            models = [str(item['id']) for item in data]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            logger.error(f'Failed to fetch chat models from {base_url}/models: {exc}')
            models = []
        self._cached_models = models
        return models
