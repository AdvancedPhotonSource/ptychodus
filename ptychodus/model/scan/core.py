from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Optional
import logging

import numpy

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from ...api.scan import (ScanDictionary, ScanFileReader, ScanFileWriter, ScanPoint,
                         ScanPointSequence, SimpleScanDictionary)
from ...api.settings import SettingsRegistry
from ...api.tree import SimpleTreeNode
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
        fileType = self._fileReaderChooser.getCurrentSimpleName()
        logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
        reader = self._fileReaderChooser.getCurrentStrategy()
        namedSequenceDict = reader.read(filePath)
        initializerList: list[TabularScanInitializer] = list()

        for name, pointSequence in namedSequenceDict.items():
            pointList = [point for point in pointSequence]
            fileInfo = ScanFileInfo(fileType, filePath, name)
            initializer = self.createTabularInitializer(pointList, fileInfo)
            initializerList.append(initializer)

        return initializerList

    def openScan(self, filePath: Path, fileFilter: str) -> list[TabularScanInitializer]:
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        return self._readScan(filePath)

    def openScanFromSettings(self) -> list[TabularScanInitializer]:
        fileInfo = ScanFileInfo.createFromSettings(self._settings)
        self._fileReaderChooser.setFromSimpleName(fileInfo.fileType)
        return self._readScan(fileInfo.filePath)

    def createRasterInitializer(self) -> CartesianScanInitializer:
        parameters = self.createInitializerParameters()
        return CartesianScanInitializer.createFromSettings(parameters, self._settings, snake=False)

    def createSnakeInitializer(self) -> CartesianScanInitializer:
        parameters = self.createInitializerParameters()
        return CartesianScanInitializer.createFromSettings(parameters, self._settings, snake=True)

    def createSpiralInitializer(self) -> SpiralScanInitializer:
        parameters = self.createInitializerParameters()
        return SpiralScanInitializer.createFromSettings(parameters, self._settings)


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
            name = initializer.name

        initializerName = name
        index = 0

        while initializerName in self._initializers:
            index += 1
            initializerName = f'{name}-{index}'

        self._initializers[initializerName] = initializer
        self.notifyObservers()

    def removeScan(self, name: str) -> None:
        if len(self._initializers) > 1:
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

    def __getitem__(self, index: int) -> ScanPoint:
        return self._initializer[index]

    def __len__(self) -> int:
        return len(self._initializer)

    def _syncFromSettings(self) -> None:
        initializerName = self._settings.initializer.value
        name = initializerName.casefold()

        # TODO confirm that this method does the right thing when rerun
        if name == 'fromfile':
            tabularList = self._initializerFactory.openScanFromSettings()

            for tabular in tabularList:
                self._repository.insertScan(tabular)

        elif name == 'raster':
            raster = self._initializerFactory.createRasterInitializer()
            self._repository.insertScan(raster)
        elif name == 'snake':
            snake = self._initializerFactory.createSnakeInitializer()
            self._repository.insertScan(snake)
        elif name == 'spiral':
            spiral = self._initializerFactory.createSpiralInitializer()
            self._repository.insertScan(spiral)
        else:
            logger.error(f'Unknown scan initializer \"{initializerName}\"!')

        self.setActive(initializerName)

    def _syncToSettings(self) -> None:
        self._initializer.syncToSettings(self._settings)

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self._syncFromSettings()


class ScanPresenter(Observable, Observer):

    @staticmethod
    def createRoot() -> SimpleTreeNode:
        return SimpleTreeNode.createRoot(['Position Data'])

    def __init__(self, initializerFactory: ScanInitializerFactory, repository: ScanRepository,
                 scan: Scan, fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._initializerFactory = initializerFactory
        self._repository = repository
        self._scan = scan
        self._fileWriterChooser = fileWriterChooser
        self._scanTree = ScanPresenter.createRoot()

    @classmethod
    def createInstance(cls, initializerFactory: ScanInitializerFactory, repository: ScanRepository, scan: Scan,
                       fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanPresenter:
        presenter = cls(initializerFactory, repository, scan, fileWriterChooser)
        presenter._updateScanTree()
        repository.addObserver(presenter)
        scan.addObserver(presenter)
        return presenter

    def getScanTree(self) -> SimpleTreeNode:
        return self._scanTree

    def _updateScanTree(self) -> None:
        nameDict: defaultdict[str, list[str]] = defaultdict(list)

        for name, initializer in self._repository.items():
            nameDict[initializer.category].append(name)

        scanTree = ScanPresenter.createRoot()

        for category, nameList in sorted(nameDict.items()):
            categoryNode = scanTree.createChild([category])

            for name in sorted(nameList):
                categoryNode.createChild([name])

        self._scanTree = scanTree
        self.notifyObservers()

    def getActiveScanPointList(self) -> list[ScanPoint]:
        return [point for point in self._scan]

    def getActiveScan(self) -> str:
        return self._scan.name

    def setActiveScan(self, name: str) -> None:
        self._scan.setActive(name)

    def insertRasterScan(self) -> CartesianScanInitializer:
        initializer = self._initializerFactory.createRasterInitializer()
        self._repository.insertScan(initializer)
        return initializer

    def insertSnakeScan(self) -> CartesianScanInitializer:
        initializer = self._initializerFactory.createSnakeInitializer()
        self._repository.insertScan(initializer)
        return initializer

    def insertSpiralScan(self) -> SpiralScanInitializer:
        initializer = self._initializerFactory.createSpiralInitializer()
        self._repository.insertScan(initializer)
        return initializer

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializerFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._initializerFactory.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        initializerList = self._initializerFactory.openScan(filePath, fileFilter)

        for initializer in initializerList:
            fileInfo = initializer.fileInfo
            name = fileInfo.fileSeriesKey if fileInfo else initializer.name
            self._repository.insertScan(initializer, name)

    def removeScan(self, name: str) -> None:
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

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self._updateScanTree()
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
        self.scanPresenter = ScanPresenter.createInstance(self._initializerFactory,
                                                          self._repository, self.scan,
                                                          fileWriterChooser)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self.scanPresenter.openScan(filePath, fileFilter)
