from __future__ import annotations
from collections.abc import Iterator
import logging

from ...api.observer import Observable, Observer
from ...api.scan import Scan, ScanPoint
from .cartesian import SnakeScanRepositoryItem
from .itemFactory import ScanRepositoryItemFactory
from .itemRepository import ScanRepository, ScanRepositoryItem, TransformedScanRepositoryItem
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class ActiveScan(Scan, Observer):

    def __init__(self, settings: ScanSettings, factory: ScanRepositoryItemFactory,
                 repository: ScanRepository, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._factory = factory
        self._repository = repository
        self._reinitObservable = reinitObservable
        self._item: TransformedScanRepositoryItem = factory.createItem('Snake')[0]
        self._name = str()

    @classmethod
    def createInstance(cls, settings: ScanSettings, factory: ScanRepositoryItemFactory,
                       repository: ScanRepository, reinitObservable: Observable) -> ActiveScan:
        scan = cls(settings, factory, repository, reinitObservable)
        scan._syncFromSettings()
        repository.addObserver(scan)
        reinitObservable.addObserver(scan)
        return scan

    @property
    def name(self) -> str:
        return self._name

    def canActivateScan(self, name: str) -> bool:
        item = self._repository.get(name)
        return False if item is None else item.canActivate

    def setActiveScan(self, name: str) -> None:
        if self._name == name:
            return

        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        if not item.canActivate:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        self._item.removeObserver(self)
        self._item = item
        self._name = name
        self._item.addObserver(self)

        self._syncToSettings()
        self.notifyObservers()

    @property
    def untransformed(self) -> Scan:
        return self._item.untransformed

    def __iter__(self) -> Iterator[int]:
        return iter(self._item)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._item[index]

    def __len__(self) -> int:
        return len(self._item)

    def _recoverIfActiveScanRemovedFromRepository(self) -> None:
        if self._name not in self._repository:
            self.setActiveScan(next(iter(self._repository)))

    def _syncFromSettings(self) -> None:
        initializerName = self._settings.initializer.value

        for item in self._factory.createItem(initializerName):
            self._repository.insertItem(item)

        self.setActiveScan(self._settings.activeScan.value)

    def _syncToSettings(self) -> None:
        self._settings.activeScan.value = self._item.name
        self._settings.initializer.value = self._item.initializer
        self._item.syncToSettings(self._settings)

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncToSettings()
        elif observable is self._repository:
            self._recoverIfActiveScanRemovedFromRepository()
        elif observable is self._reinitObservable:
            self._syncFromSettings()
