import logging

from ..itemRepository import RepositoryItemSettingsDelegate, SelectedRepositoryItem
from .factory import ScanRepositoryItemFactory
from .repository import ScanRepository, ScanRepositoryItem
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class ScanRepositoryItemSettingsDelegate(RepositoryItemSettingsDelegate[ScanRepositoryItem]):

    def __init__(self, settings: ScanSettings, factory: ScanRepositoryItemFactory,
                 repository: ScanRepository) -> None:
        super().__init__()
        self._settings = settings
        self._factory = factory
        self._repository = repository

    def syncFromSettings(self) -> str:
        initializerName = self._settings.initializer.value
        item = self._factory.createItem(initializerName)

        if item is None:
            logger.error(f'Unknown scan initializer \"{initializerName}\"!')
        else:
            itemName = self._repository.insertItem(item)

        return itemName

    def syncToSettings(self, item: ScanRepositoryItem) -> None:
        itemInitializer = item.getInitializer()

        if itemInitializer is None:
            raise RuntimeError('Unable to sync item to settings without initializer!')

        self._settings.initializer.value = itemInitializer.simpleName
        itemInitializer.syncToSettings(self._settings)
        item.syncToSettings(self._settings)


SelectedScan = SelectedRepositoryItem[ScanRepositoryItem]
