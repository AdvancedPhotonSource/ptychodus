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

    def syncFromSettings(self) -> str:
        initializerName = self._settings.initializer.value
        item = self._factory.createItem(initializerName)
        itemName = str()

        if item is None:
            logger.error(f'Unknown object initializer \"{initializerName}\"!')
        else:
            itemName = self._repository.insertItem(item)

        return itemName

    def syncToSettings(self, item: ObjectRepositoryItem) -> None:
        itemInitializer = item.getInitializer()

        if itemInitializer is None:
            raise RuntimeError('Unable to sync item to settings without initializer!')

        self._settings.initializer.value = itemInitializer.simpleName
        itemInitializer.syncToSettings(self._settings)


SelectedObject = SelectedRepositoryItem[ObjectRepositoryItem]
