from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, Iterator, Optional, Union
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import (ScanFileReader, ScanFileWriter, ScanPoint, ScanPointSequence,
                         SimpleScanDictionary)
from ...api.settings import SettingsRegistry
from .cartesian import CartesianScanInitializer
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings
from .spiral import SpiralScanInitializer
from .tabular import ScanFileInfo, TabularScanInitializer

logger = logging.getLogger(__name__)


class ScanInitializerFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser

    def createInitializerParameters(self) -> ScanInitializerParameters:
        return ScanInitializerParameters.createFromSettings(self._rng, self._settings)

    def createTabularInitializer(self, pointList: list[ScanPoint],
                                 fileInfo: Optional[ScanFileInfo]) -> TabularScanInitializer:
        parameters = self.createInitializerParameters()
        return TabularScanInitializer(parameters, pointList, fileInfo)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _readScan(self, filePath: Path) -> list[TabularScanInitializer]:
        initializerList: list[TabularScanInitializer] = list()

        if filePath is not None and filePath.is_file():
            fileType = self._fileReaderChooser.getCurrentSimpleName()
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            reader = self._fileReaderChooser.getCurrentStrategy()
            namedSequenceDict = reader.read(filePath)

            for name, pointSequence in namedSequenceDict.items():
                pointList = [point for point in pointSequence]
                fileInfo = ScanFileInfo(fileType, filePath, name)
                initializer = self.createTabularInitializer(pointList, fileInfo)
                initializerList.append(initializer)
        else:
            logger.debug(f'Refusing to read invalid file path {filePath}')

        return initializerList

    def openScan(self, filePath: Path, fileFilter: str) -> list[TabularScanInitializer]:
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        return self._readScan(filePath)

    def openScanFromSettings(self) -> list[TabularScanInitializer]:
        fileInfo = ScanFileInfo.createFromSettings(self._settings)
        self._fileReaderChooser.setFromSimpleName(fileInfo.fileType)
        return self._readScan(fileInfo.filePath)

    def getInitializerNameList(self) -> list[str]:
        return ['Raster', 'Snake', 'Spiral']

    def createInitializer(self, name: str) -> Optional[ScanInitializer]:
        nameLower = name.casefold()
        parameters = self.createInitializerParameters()

        if nameLower == 'raster':
            return CartesianScanInitializer.createFromSettings(parameters,
                                                               self._settings,
                                                               snake=False)
        elif nameLower == 'snake':
            return CartesianScanInitializer.createFromSettings(parameters,
                                                               self._settings,
                                                               snake=True)
        elif nameLower == 'spiral':
            return SpiralScanInitializer.createFromSettings(parameters, self._settings)

        return None


class ScanRepository(Mapping[str, ScanInitializer], Observable):

    def __init__(self) -> None:
        super().__init__()
        self._initializers: dict[str, ScanInitializer] = dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self._initializers)

    def __getitem__(self, name: str) -> ScanInitializer:
        return self._initializers[name]

    def __len__(self) -> int:
        return len(self._initializers)

    def insertScan(self, initializer: ScanInitializer, name: Optional[str] = None) -> None:
        if name is None:
            name = initializer.variant

        initializerName = name
        index = 0

        while initializerName in self._initializers:
            index += 1
            initializerName = f'{name}-{index}'

        self._initializers[initializerName] = initializer
        self.notifyObservers()

    def canRemoveScan(self, name: str) -> bool:
        return len(self._initializers) > 1

    def removeScan(self, name: str) -> None:
        if self.canRemoveScan(name):
            try:
                self._initializers.pop(name)
            except KeyError:
                pass

        self.notifyObservers()


class Scan(ScanPointSequence, Observable, Observer):

    def __init__(self, settings: ScanSettings, initializerFactory: ScanInitializerFactory,
                 repository: ScanRepository, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._initializerFactory = initializerFactory
        self._repository = repository
        self._reinitObservable = reinitObservable
        self._initializer: ScanInitializer = initializerFactory.createTabularInitializer([], None)
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

    def setActive(self, name: str) -> None:
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

        # FIXME confirm that this method does the right thing when rerun
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

        self.setActive(initializerName)

    def _syncToSettings(self) -> None:
        self._initializer.syncToSettings(self._settings)

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self._syncFromSettings()


@dataclass(frozen=True)
class ScanRepositoryEntry:
    name: str
    category: str
    variant: str
    pointSequence: ScanPointSequence


class ScanPresenter(Observable, Observer):

    def __init__(self, initializerFactory: ScanInitializerFactory, repository: ScanRepository,
                 scan: Scan, fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._initializerFactory = initializerFactory
        self._repository = repository
        self._scan = scan
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(cls, initializerFactory: ScanInitializerFactory, repository: ScanRepository,
                       scan: Scan,
                       fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanPresenter:
        presenter = cls(initializerFactory, repository, scan, fileWriterChooser)
        repository.addObserver(presenter)
        scan.addObserver(presenter)
        return presenter

    def getScanRepositoryContents(self) -> list[ScanRepositoryEntry]:
        return [
            ScanRepositoryEntry(name, ini.category, ini.variant, ini)
            for name, ini in self._repository.items()
        ]

    def getActiveScanPointList(self) -> list[ScanPoint]:
        return [point for point in self._scan]

    def getActiveScan(self) -> str:
        return self._scan.name

    def setActiveScan(self, name: str) -> None:
        self._scan.setActive(name)

    def getInitializerNameList(self) -> list[str]:
        return self._initializerFactory.getInitializerNameList()

    def initializeScan(self, name: str) -> None:
        initializer = self._initializerFactory.createInitializer(name)

        if initializer is None:
            logger.error(f'Unknown scan initializer \"{name}\"!')
        else:
            self._repository.insertScan(initializer)

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializerFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._initializerFactory.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        initializerList = self._initializerFactory.openScan(filePath, fileFilter)

        for initializer in initializerList:
            fileInfo = initializer.fileInfo
            name = fileInfo.fileSeriesKey if fileInfo else initializer.variant
            self._repository.insertScan(initializer, name)

    def canRemoveScan(self, name: str) -> bool:
        return self._repository.canRemoveScan(name) and name != self._scan.name

    def removeScan(self, name: str) -> None:
        if self.canRemoveScan(name):
            self._repository.removeScan(name)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveScan(self, filePath: Path, fileFilter: str, name: str) -> None:
        try:
            initializer = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, SimpleScanDictionary.createFromUnnamedSequence(initializer))

        # FIXME if strategy is Tabular and no fileInfo, add fileInfo to
        #       strategy and make sure that strategy calls notifyObservers

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()


class ScanCore:

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._repository = ScanRepository()
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._initializerFactory = ScanInitializerFactory(rng, self._settings, fileReaderChooser)
        self.scan = Scan.createInstance(self._settings, self._initializerFactory, self._repository,
                                        settingsRegistry)
        self.presenter = ScanPresenter.createInstance(self._initializerFactory, self._repository,
                                                      self.scan, fileWriterChooser)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self.presenter.openScan(filePath, fileFilter)
