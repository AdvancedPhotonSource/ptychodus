import logging

import httpx

from .endpoint import get_argo_user, get_gateway_url
from .settings import AgentSettings

logger = logging.getLogger(__name__)


class ModelCatalog:
    """Fetches and caches the list of chat models advertised by the configured gateway."""

    def __init__(self, settings: AgentSettings) -> None:
        self._settings = settings
        self._cached_models: list[str] | None = None

    def get_available_models(self) -> list[str]:
        if self._cached_models is None:
            return self.refresh()
        return self._cached_models

    def refresh(self) -> list[str]:
        models_url = f'{get_gateway_url(self._settings.base_url.get_value())}/v1/models'
        try:
            headers = {'Authorization': f'Bearer {get_argo_user(models_url)}'}
            response = httpx.get(models_url, timeout=10.0, headers=headers)
            response.raise_for_status()
            data = response.json()['data']
            models = [str(item['id']) for item in data]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
            logger.error(f'Failed to fetch chat models from {models_url}: {exc}')
            models = []
        self._cached_models = models
        return models
