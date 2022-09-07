from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import overload, Union
import logging

from ...api.observer import Observable, Observer
from ...api.scan import ScanPoint, ScanPointSequence
from .initializer import ScanInitializer
from .initializerFactory import ScanInitializerFactory
from .repository import ScanRepository
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class Scan(ScanPointSequence, Observable, Observer):

    def __init__(self, settings: ScanSettings, initializerFactory: ScanInitializerFactory,
                 repository: ScanRepository, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._initializerFactory = initializerFactory
        self._repository = repository
        self._reinitObservable = reinitObservable
        self._initializer: ScanInitializer = initializerFactory.createTabularInitializer([], '',
                                                                                         None)
        self._name = str()

    @classmethod
    def createInstance(cls, settings: ScanSettings, initializerFactory: ScanInitializerFactory,
                       repository: ScanRepository, reinitObservable: Observable) -> Scan:
        scan = cls(settings, initializerFactory, repository, reinitObservable)
        scan._syncFromSettings()
        reinitObservable.addObserver(scan)
        return scan

    @property
    def name(self) -> str:
        return self._name

    def setActiveScan(self, name: str) -> None:
        if self._name == name:
            return

        try:
            initializer = self._repository[name]
        except KeyError:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        self._initializer.removeObserver(self)
        self._initializer = initializer
        self._name = name
        self._initializer.addObserver(self)

        self._syncToSettings()
        self.notifyObservers()

    @overload
    def __getitem__(self, index: int) -> ScanPoint:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ScanPoint]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[ScanPoint, Sequence[ScanPoint]]:
        return self._initializer[index]

    def __len__(self) -> int:
        return len(self._initializer)

    def _syncFromSettings(self) -> None:
        initializerName = self._settings.initializer.value
        name = initializerName.casefold()

        if name == 'fromfile':
            tabularList = self._initializerFactory.openScanFromSettings()

            for tabular in tabularList:
                self._repository.insertScan(tabular)
        else:
            initializer = self._initializerFactory.createInitializer(name)

            if initializer is None:
                logger.error(f'Unknown scan initializer \"{initializerName}\"!')
            else:
                self._repository.insertScan(initializer)

        self.setActiveScan(self._settings.activeScan.value)

    def _syncToSettings(self) -> None:
        self._initializer.syncToSettings(self._settings)
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncToSettings()
        elif observable is self._reinitObservable:
            self._syncFromSettings()
