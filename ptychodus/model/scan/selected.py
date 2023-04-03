from ..itemRepository import RepositoryItemSettingsDelegate, SelectedRepositoryItem
from .factory import ScanRepositoryItemFactory
from .repository import ScanRepository, ScanRepositoryItem, TransformedScanRepositoryItem
from .settings import ScanSettings


class ScanRepositoryItemSettingsDelegate(
        RepositoryItemSettingsDelegate[TransformedScanRepositoryItem]):

    def __init__(self, settings: ScanSettings, factory: ScanRepositoryItemFactory,
                 repository: ScanRepository) -> None:
        super().__init__()
        self._settings = settings
        self._factory = factory
        self._repository = repository

    def syncFromSettings(self) -> str:
        initializerName = self._settings.initializer.value

        for item in self._factory.createItem(initializerName):
            self._repository.insertItem(item)

        return self._settings.activeScan.value

    def syncToSettings(self, item: ScanRepositoryItem) -> None:
        self._settings.initializer.value = item.initializer
        self._settings.activeScan.value = item.nameHint
        item.syncToSettings(self._settings)


SelectedScan = SelectedRepositoryItem[TransformedScanRepositoryItem]
