import logging

from ..itemRepository import RepositoryItemSettingsDelegate, SelectedRepositoryItem
from .factory import ProbeRepositoryItemFactory
from .repository import ProbeRepository, ProbeRepositoryItem
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeRepositoryItemSettingsDelegate(RepositoryItemSettingsDelegate[ProbeRepositoryItem]):

    def __init__(self, settings: ProbeSettings, factory: ProbeRepositoryItemFactory,
                 repository: ProbeRepository) -> None:
        super().__init__()
        self._settings = settings
        self._factory = factory
        self._repository = repository

    def syncFromSettings(self) -> str | None:
        name = self._settings.initializer.value
        item = self._factory.createItemFromSimpleName(name)

        if item is None:
            logger.error('Failed to create item!')
            return None

        itemInitializer = item.getInitializer()

        if itemInitializer is None:
            raise RuntimeError('Unable to sync item from settings without initializer!')

        itemInitializer.syncFromSettings(self._settings)
        item.syncFromSettings(self._settings)

        return self._repository.insertItem(item)

    def syncToSettings(self, item: ProbeRepositoryItem) -> None:
        itemInitializer = item.getInitializer()

        if itemInitializer is None:
            logger.warn('Unable to sync item to settings without initializer!')
        else:
            self._settings.initializer.value = itemInitializer.simpleName
            itemInitializer.syncToSettings(self._settings)
            item.syncToSettings(self._settings)


SelectedProbe = SelectedRepositoryItem[ProbeRepositoryItem]
