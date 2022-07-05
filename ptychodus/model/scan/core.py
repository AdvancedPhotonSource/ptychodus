from __future__ import annotations
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
from ...api.settings import SettingsRegistry, SettingsGroup
from .initializer import ScanInitializer
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class NullScanInitializer(ScanInitializer):
    def __init__(self, rng: numpy.random.Generator = numpy.random.default_rng()) -> None:
        super().__init__(ScanInitializerParameters(rng))
        self._pointList: list[ScanPoint] = list()

    @property
    def category(self) -> str:
        return 'Null'

    @property
    def name(self) -> str:
        return 'Null'

    def _getPoint(self, index: int) -> ScanPoint:
        return self._pointList[index]

    def __len__(self) -> int:
        return len(self._pointList)


class ScanInitializerRepository(Mapping[str, Mapping[str, ScanInitializer]], Observable):

    def __init__(self, fileReaderChooser: PluginChooser[ScanFileReader], fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._initializers: dict[str, dict[str, ScanInitializer]] = dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self._initializers)

    def __getitem__(self, category: str) -> Mapping[str, ScanInitializer]:
        return self._initializers[category]

    def __len__(self) -> int:
        return len(self._initializers)

    def insert(self, initializer: ScanInitializer) -> None:
        if initializer.category not in self._initializers:
            self._initializers[initializer.category] = dict()

        initializerCategory = self._initializers[initializer.category]
        name = initializer.name
        index = 0

        while name in initializerCategory:
            index += 1
            name = f'{initializer.name}-{index}'

        initializerCategory[name] = initializer
        self.notifyObservers()

    def remove(self, category: str, name: str) -> None:
        if category in self._initializers:
            initializerCategory = self._initializers[category]

            try:
                initializerCategory.pop(name)
            except KeyError:
                pass

            if not initializerCategory:
                self._initializers.pop(category)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        logger.debug(f'Reading \"{filePath}\" as \"{fileFilter}\"')
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        reader = self._fileReaderChooser.getCurrentStrategy()
        scanDict = reader.read(filePath)

        rng = None # FIXME solve by createing ScanInitializerFactory that has rng?
        fileType = self._fileReaderChooser.getCurrentSimpleName()

        for name, pointSequence in scanDict.items():
            parameters = ScanInitializerParameters(rng)
            pointList = [point for point in pointSequence]
            fileInfo = ScanFileInfo(fileType, filePath, name)
            initializer = TabularScanInitializer(parameters, name, pointList, fileInfo)
            self.insert(initializer)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveScan(self, filePath: Path, fileFilter: str, category: str, initializer: str) -> None:
        try:
            initializer = self._initializers[category][initializer]
        except KeyError:
            logger.error(f'Unable to locate \"{category}/{initializer}\"!')
            return

        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, SimpleScanDictionary.createFromUnnamedSequence(initializer))


class Scan(ScanPointSequence, Observable, Observer):

    def __init__(self, settings: ScanSettings, initializerRepository: ScanInitializerRepository,
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._initializerRepository = initializerRepository
        self._reinitObservable = reinitObservable
        self._initializer: ScanInitializer = NullScanInitializer()

    @classmethod
    def createInstance(cls, settings: ScanSettings,
                       initializerRepository: ScanInitializerRepository,
                       reinitObservable: Observable) -> Scan:
        scan = cls(settings, initializerRepository, fileReaderChooser, reinitObservable)
        scan._syncActiveFromSettings()
        reinitObservable.addObserver(scan)
        return scan

    def setActive(self, category: str, initializer: str) -> None:
        try:
            self._initializer = self._initializerRepository[category][initializer]
        except KeyError:
            logger.error(f'Failed to activate \"{category}/{initializer}\"!')
            return

        self._syncActiveToSettings()
        self.notifyObservers()

    def _syncActiveFromSettings(self) -> None:
        # FIXME make first added initializer active; notifyObservers when initializer changes
        # FIXME create and add to initializerRepository; notifyObservers when changed
        pass

    def _syncActiveToSettings(self) -> None:
        self._initializer.syncToSettings(self._settings)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._initializer[index]

    def __len__(self) -> int:
        return len(self._initializer)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncActiveFromSettings()


class ScanPresenter(Observable, Observer):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: ScanSettings, scan: Scan, initializer: ScanInitializer) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan
        self._initializer = initializer

    @classmethod
    def createInstance(cls, settings: ScanSettings, scan: Scan,
                       initializer: ScanInitializer) -> ScanPresenter:
        presenter = cls(settings, scan, initializer)
        settings.addObserver(presenter)
        scan.addObserver(presenter)
        initializer.addObserver(presenter)
        return presenter

    def getInitializerList(self) -> list[str]:
        return self._initializer.getInitializerList()

    def getInitializer(self) -> str:
        return self._initializer.getInitializer()

    def setInitializer(self, name: str) -> None:
        self._initializer.setInitializer(name)

    def initializeScan(self) -> None:
        self._initializer.initializeScan()

    def getTransformList(self) -> list[str]:
        return self._scan.getTransformList()

    def getTransform(self) -> str:
        return self._scan.getTransform()

    def setTransform(self, name: str) -> None:
        self._scan.setTransform(name)

    def getScanPointList(self) -> list[ScanPoint]:
        return [scanPoint for scanPoint in self._scan]

    def getNumberOfScanPointsLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getNumberOfScanPoints(self) -> int:
        limits = self.getNumberOfScanPointsLimits()
        return limits.clamp(len(self._scan))

    def getExtentXLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getExtentX(self) -> int:
        limits = self.getExtentXLimits()
        return limits.clamp(self._settings.extentX.value)

    def setExtentX(self, value: int) -> None:
        self._settings.extentX.value = value

    def getExtentYLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getExtentY(self) -> int:
        limits = self.getExtentYLimits()
        return limits.clamp(self._settings.extentY.value)

    def setExtentY(self, value: int) -> None:
        self._settings.extentY.value = value

    def getStepSizeXInMeters(self) -> Decimal:
        return self._settings.stepSizeXInMeters.value

    def setStepSizeXInMeters(self, value: Decimal) -> None:
        self._settings.stepSizeXInMeters.value = value

    def getStepSizeYInMeters(self) -> Decimal:
        return self._settings.stepSizeYInMeters.value

    def setStepSizeYInMeters(self, value: Decimal) -> None:
        self._settings.stepSizeYInMeters.value = value

    def getJitterRadiusInMeters(self) -> Decimal:
        return self._settings.jitterRadiusInMeters.value

    def setJitterRadiusInMeters(self, value: Decimal) -> None:
        self._settings.jitterRadiusInMeters.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._initializer:
            self.notifyObservers()


class ScanCore:

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._initializerRepository = ScanInitializerRepository(fileReaderChooser, fileWriterChooser)
        self.scan = Scan.createInstance(self._settings)
        self._fileScanInitializer = FileScanInitializer.createInstance(
            self._settings, fileReaderChooser)
        self._scanInitializer = ScanInitializer.createInstance(
            self.rng, self._settings, self._scan, self._fileScanInitializer,
            fileWriterChooser, settingsRegistry)
        self.scanPresenter = ScanPresenter.createInstance(self._settings, self._scan,
                                                          self._scanInitializer)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self.scanPresenter.openScan(filePath, fileFilter)
