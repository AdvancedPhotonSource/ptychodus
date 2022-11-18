from __future__ import annotations
from collections.abc import Iterator
import logging

from ...api.observer import Observable, Observer
from ...api.scan import Scan, ScanPoint
from .cartesian import CartesianScanRepositoryItem
from .factory import ScanRepositoryItemFactory
from .repository import ScanRepository
from .repositoryItem import ScanRepositoryItem
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class ActiveScan(Scan, Observer):

    def __init__(self, settings: ScanSettings, initializerFactory: ScanRepositoryItemFactory,
                 repository: ScanRepository, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._initializerFactory = initializerFactory
        self._repository = repository
        self._reinitObservable = reinitObservable
        self._initializer: ScanRepositoryItem = CartesianScanRepositoryItem(snake=True)
        self._name = str()

    @classmethod
    def createInstance(cls, settings: ScanSettings, initializerFactory: ScanRepositoryItemFactory,
                       repository: ScanRepository, reinitObservable: Observable) -> ActiveScan:
        scan = cls(settings, initializerFactory, repository, reinitObservable)
        scan._syncFromSettings()
        reinitObservable.addObserver(scan)
        return scan

    @property
    def name(self) -> str:
        return self._name

    def canActivateScan(self, name: str) -> bool:
        item = self._repository.get(name)

        if item is not None:
            return item.canActivate

        return False

    def setActiveScan(self, name: str) -> None:
        if self._name == name:
            return

        try:
            initializer = self._repository[name]
        except KeyError:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        if not initializer.canActivate:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        self._initializer.removeObserver(self)
        self._initializer = initializer
        self._name = name
        self._initializer.addObserver(self)

        self._syncToSettings()
        self.notifyObservers()

    def __iter__(self) -> Iterator[int]:
        return iter(self._initializer)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._initializer[index]

    def __len__(self) -> int:
        return len(self._initializer)

    def _syncFromSettings(self) -> None:
        initializerName = self._settings.initializer.value
        name = initializerName.casefold()

        if name == 'fromfile':
            tabularList = self._initializerFactory.openScanFromSettings()

            for tabular in tabularList:
                self._repository.insertItem(tabular)
        else:
            initializer = self._initializerFactory.createInitializer(name)

            if initializer is None:
                logger.error(f'Unknown scan initializer \"{initializerName}\"!')
            else:
                self._repository.insertItem(initializer)

        self.setActiveScan(self._settings.activeScan.value)

    def _syncToSettings(self) -> None:
        self._settings.initializer.value = self._initializer.name
        self._initializer.syncToSettings(self._settings)
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncToSettings()
        elif observable is self._reinitObservable:
            self._syncFromSettings()
