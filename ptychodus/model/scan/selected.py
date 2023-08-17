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

    def syncFromSettings(self) -> str | None:
        name = self._settings.initializer.value
        item = self._factory.createItemFromInitializerName(name)

        if item is None:
            logger.error('Failed to create item!')
            return None

        itemInitializer = item.getInitializer()

        if itemInitializer is None:
            raise RuntimeError('Unable to sync item from settings without initializer!')

        itemInitializer.syncFromSettings(self._settings)
        item.syncFromSettings(self._settings)

        return self._repository.insertItem(item)

    def syncToSettings(self, item: ScanRepositoryItem) -> None:
        itemInitializer = item.getInitializer()

        if itemInitializer is None:
            logger.warn('Unable to sync item to settings without initializer!')
        else:
            self._settings.initializer.value = itemInitializer.simpleName
            itemInitializer.syncToSettings(self._settings)
            item.syncToSettings(self._settings)


SelectedScan = SelectedRepositoryItem[ScanRepositoryItem]
