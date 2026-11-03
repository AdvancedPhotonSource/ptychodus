import logging

import httpx

from ptychodus.api.language_model import LanguageModelProviderError

from .provider import LanguageModelProviderSelection

logger = logging.getLogger(__name__)


class ModelCatalog:
    """Fetches and caches the chat models each provider advertises.

    Cached per provider, so switching providers neither shows another's list nor
    fetches again; :meth:`refresh` re-fetches the selected provider's.
    """

    def __init__(self, selection: LanguageModelProviderSelection) -> None:
        self._selection = selection
        self._cached_models: dict[str, list[str]] = {}

    def get_available_models(self) -> list[str]:
        name = self._selection.get_current().plugin.simple_name
        models = self._cached_models.get(name)
        return self.refresh() if models is None else models

    def refresh(self) -> list[str]:
        options = self._selection.get_current()
        name = options.plugin.simple_name

        try:
            models = list(options.list_models())
        except (httpx.HTTPError, LanguageModelProviderError) as exc:
            logger.error(f'Failed to fetch {options.plugin.display_name} models: {exc}')
            models = []

        self._cached_models[name] = models
        return models
