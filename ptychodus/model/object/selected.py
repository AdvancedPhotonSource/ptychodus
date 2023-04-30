import logging

from ..itemRepository import RepositoryItemSettingsDelegate, SelectedRepositoryItem
from .factory import ObjectRepositoryItemFactory
from .repository import ObjectRepository, ObjectRepositoryItem
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectRepositoryItemSettingsDelegate(RepositoryItemSettingsDelegate[ObjectRepositoryItem]):

    def __init__(self, settings: ObjectSettings, factory: ObjectRepositoryItemFactory,
                 repository: ObjectRepository) -> None:
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

        return self._repository.insertItem(item)

    def syncToSettings(self, item: ObjectRepositoryItem) -> None:
        itemInitializer = item.getInitializer()

        if itemInitializer is None:
            raise RuntimeError('Unable to sync item to settings without initializer!')

        self._settings.initializer.value = itemInitializer.simpleName
        itemInitializer.syncToSettings(self._settings)


SelectedObject = SelectedRepositoryItem[ObjectRepositoryItem]
