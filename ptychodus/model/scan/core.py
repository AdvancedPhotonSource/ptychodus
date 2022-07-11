from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Iterator
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
from .tabular import TabularScanInitializer

logger = logging.getLogger(__name__)


class ScanInitializerFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings) -> None:
        self._rng = rng
        self._settings = settings

    def createRasterInitializer(self) -> CartesianScanInitializer:
        parameters = self.createInitializerParameters()
        return CartesianScanInitializer.createFromSettings(parameters, self._settings, snake=False)

    def createSnakeInitializer(self) -> CartesianScanInitializer:
        parameters = self.createInitializerParameters()
        return CartesianScanInitializer.createFromSettings(parameters, self._settings, snake=True)

    def createSpiralInitializer(self) -> SprialScanInitializer:
        parameters = self.createInitializerParameters()
        return SpiralScanInitializer.createFromSettings(parameters, self._settings)

    def createTabularInitializer(self, pointList: list[ScanPoint],
                                 fileInfo: Optional[ScanFileInfo]) -> TabularScanInitializer:
        parameters = self.createInitializerParameters()
        return TabularScanInitializer(parameters, pointList, fileInfo)

    def createInitializerParameters(self) -> ScanInitializerParameters:
        return ScanInitializerParameters.createFromSettings(self._rng, self._settings)


class ScanInitializerRepository(Mapping[str, ScanInitializer], Observable, Observer):

    def __init__(self, initializerFactory: ScanInitializerFactory,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._initializerFactory = initializerFactory
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable
        self._initializers: dict[str, ScanInitializer] = dict()

    @classmethod
    def createInstance(self, initializerFactory: ScanInitializerFactory,
                       fileReaderChooser: PluginChooser[ScanFileReader],
                       fileWriterChooser: PluginChooser[ScanFileWriter],
                       reinitObservable: Observable) -> ScanInitializerRepository:
        repository = cls(initializerFactory, fileReaderChooser, fileWriterChooser,
                         reinitObservable)
        repository._syncFromSettings()
        reinitObservable.addObserver(repository)
        return repository

    def getDefaultInitializer(self) -> ScanInitializer:
        return list(self._initializers.values())[0] if len(self._initializers) > 0 \
                else self._initializerFactory.createTabularInitializer([], None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._initializers)

    def __getitem__(self, name: str) -> ScanInitializer:
        return self._initializers[name]

    def __len__(self) -> int:
        return len(self._initializers)

    def insert(self, initializer: ScanInitializer, name: Optional[str] = None) -> None:
        if name is None:
            name = initializer.name

        initializerName = name
        index = 0

        while initializerName in self._initializers:
            index += 1
            initializerName = f'{name}-{index}'

        self._initializers[initializerName] = initializer
        self.notifyObservers()

    def remove(self, name: str) -> None:
        if len(self._initializers) > 1:
            try:
                self._initializers.pop(name)
            except KeyError:
                pass

        self.notifyObservers()

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _loadScan(self, filePath: Path) -> None:
        fileType = self._fileReaderChooser.getCurrentSimpleName()
        reader = self._fileReaderChooser.getCurrentStrategy()
        scanDict = reader.read(filePath)

        for name, pointSequence in scanDict.items():
            pointList = [point for point in pointSequence]
            fileInfo = ScanFileInfo(fileType, filePath, name)
            initializer = self._initializerFactory.createTabularInitializer(pointList, fileInfo)
            self.insert(initializer, name)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        logger.debug(f'Reading \"{filePath}\" as \"{fileFilter}\"')
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        self._loadScan(filePath)

    def _openScanFromSettings(self) -> None:
        fileInfo = ScanFileInfo.createFromSettings(self._settings)
        logger.debug(f'Reading \"{fileInfo.filePath}\" as \"{fileInfo.fileType}\"')
        self._fileReaderChooser.setFromSimpleName(fileInfo.fileType)
        self._loadScan(fileInfo.filePath)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveScan(self, filePath: Path, fileFilter: str, name: str) -> None:
        try:
            initializer = self._initializers[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, SimpleScanDictionary.createFromUnnamedSequence(initializer))

    def _syncFromSettings(self) -> None:
        # FIXME this method needs to do the right thing when rerun
        initializerName = self._settings.initializer.value
        name = initializerName.casefold()

        if name == 'fromfile':
            self._openScanFromSettings()
        elif name == 'raster':
            raster = self._initializerFactory.createRasterInitializer()
            self.insert(raster)
        elif name == 'snake':
            snake = self._initializerFactory.createSnakeInitializer()
            self.insert(snake)
        elif name == 'spiral':
            spiral = self._initializerFactory.createSpiralInitializer()
            self.insert(spiral)
        else:
            logger.error(f'Unknown scan initializer \"{initializerName}\"!')

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()


class Scan(ScanPointSequence, Observable, Observer):

    def __init__(self, settings: ScanSettings, initializerRepository: ScanInitializerRepository,
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._initializerRepository = initializerRepository
        self._reinitObservable = reinitObservable
        self._initializer = initializerRepository.getDefaultInitializer()

    @classmethod
    def createInstance(cls, settings: ScanSettings,
                       initializerRepository: ScanInitializerRepository,
                       reinitObservable: Observable) -> Scan:
        scan = cls(settings, initializerRepository, reinitObservable)
        scan._syncFromSettings()
        reinitObservable.addObserver(scan)
        return scan

    def setActive(self, name: str) -> None:
        try:
            initializer = self._initializerRepository[name]
        except KeyError:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        self._initializer.removeObserver(self)
        self._initializer = initializer
        self._initializer.addObserver(self)

        self._syncToSettings()
        self.notifyObservers()

    def _syncFromSettings(self) -> None:
        self.setActive(self._settings.initializer.value)

    def _syncToSettings(self) -> None:
        self._initializer.syncToSettings(self._settings)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._initializer[index]

    def __len__(self) -> int:
        return len(self._initializer)

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self._syncFromSettings()


class ScanPresenter(Observable, Observer):

    def __init__(self, initializerRepository: ScanInitializerRepository, scan: Scan) -> None:
        super().__init__()
        self._initializerRepository = initializerRepository
        self._scan = scan

    @classmethod
    def createInstance(cls, initializerRepository: ScanInitializerRepository,
                       scan: Scan) -> ScanPresenter:
        presenter = cls(initializerRepository, scan)
        initializerRepository.addObserver(presenter)
        scan.addObserver(presenter)
        return presenter

    def getInitializerTree(self) -> SimpleTreeNode:
        return self._initializerTree

    def _updateInitializerTree(self) -> None:
        nameDict: defaultdict[str, list[str]] = defaultdict(list)

        for name, initializer in self._initializerRepository.items():
            initializerDict[initializer.category].append(name)

        initializerTree = SimpleTreeNode.createRoot(['Name'])

        for category, nameList in sorted(nameDict.items()):
            categoryNode = initializerTree.createChild([category])

            for name in sorted(nameList):
                categoryNode.createChild([name])

        self._initializerTree = initializerTree
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._initializerRepository:
            self._updateInitializerTree()
        elif observable is self._scan:
            self.notifyObservers()


class ScanCore:

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._initializerFactory = ScanInitializerFactory(rng, self._settings)
        self._initializerRepository = ScanInitializerRepository(self._initializerFactory,
                                                                fileReaderChooser,
                                                                fileWriterChooser,
                                                                settingsRegistry)
        self.scan = Scan.createInstance(self._settings, self._initializerRepository,
                                        settingsRegistry)
        self.scanPresenter = ScanPresenter.createInstance(self._initializerRepository, self.scan)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self.scanPresenter.openScan(filePath, fileFilter)
