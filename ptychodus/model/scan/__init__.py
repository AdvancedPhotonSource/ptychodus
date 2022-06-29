from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

import numpy

from ..api.geometry import Interval, Box
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry
from ..api.scan import ScanFileReader, ScanFileWriter, ScanPoint, ScanPointSequence
from ..api.settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class ScanSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Cartesian/Snake')
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'CSV')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath',
                                                           Path('/path/to/scan.csv'))
        self.extentX = settingsGroup.createIntegerEntry('ExtentX', 10)
        self.extentY = settingsGroup.createIntegerEntry('ExtentY', 10)
        self.stepSizeXInMeters = settingsGroup.createRealEntry('StepSizeXInMeters', '1e-6')
        self.stepSizeYInMeters = settingsGroup.createRealEntry('StepSizeYInMeters', '1e-6')
        self.jitterRadiusInMeters = settingsGroup.createRealEntry('JitterRadiusInMeters', '0')
        self.transform = settingsGroup.createStringEntry('Transform', '+X+Y')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ScanSettings:
        settings = cls(settingsRegistry.createGroup('Scan'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class ScanInitializerRepository(Observable, Observer):

    @staticmethod
    def _createPluginEntry(initializer: ScanInitializer) -> PluginEntry[ScanInitializer]:
        return PluginEntry[ScanInitializer](simpleName=initializer.simpleName,
                                            displayName=initializer.displayName,
                                            strategy=initializer)

    def __init__(self, settings: ScanSettings, fileReaderChooser: PluginChooser[ScanFileReader],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan
        self._reinitObservable = reinitObservable
        self._fileInitializer = FileScanInitializer.createInstance(settings, fileReaderChooser)
        self._initializerChooser = PluginChooser[ScanInitializer](
            AvailableScanDictionary._createPluginEntry(self._fileInitializer))

    @classmethod
    def createInstance(cls, settings: ScanSettings,
                       fileReaderChooser: PluginChooser[ScanFileReader],
                       reinitObservable: Observable) -> ScanInitializerRepository:
        initializer = cls(settings, fileReaderChooser, reinitObservable)

        initializer._initializerChooser.addStrategy(
            AvailableScanDictionary._createPluginEntry(CartesianScanInitializer(settings))
        initializer._initializerChooser.addStrategy(
            AvailableScanDictionary._createPluginEntry(SpiralScanInitializer(settings))

        settings.initializer.addObserver(initializer)
        initializer._initializerChooser.addObserver(initializer)
        initializer._syncInitializerFromSettings()
        reinitObservable.addObserver(initializer)

        return initializer

# FIXME BEGIN update to support category/initializer format
    def getInitializerList(self) -> list[str]: # FIXME dict[str,list[str]]
        return self._initializerChooser.getDisplayNameList()

    def getInitializer(self) -> str: # FIXME category/initializer format
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializer(self, name: str) -> None: # FIXME category/initializer format
        self._initializerChooser.setFromDisplayName(name)

    def initializeScan(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Scan')
        self._scan.setScanPoints(initializer)  # FIXME
# FIXME END

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileInitializer.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._fileInitializer.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._fileInitializer.openScan(filePath, fileFilter)
        self._initializerChooser.setToDefault()
        self.initializeScan()

    def _syncInitializerFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value) # FIXME category/initializer format

    def _syncInitializerToSettings(self) -> None:
        self._settings.initializer.value = self._initializerChooser.getCurrentSimpleName() # FIXME category/initializer format
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self._syncInitializerFromSettings()
        elif observable is self._initializerChooser:
            self._syncInitializerToSettings()
        elif observable is self._reinitObservable:
            self.initializeScan()


class ActiveScanPointSequence(ScanPointSequence, Observable):

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._fileWriterChooser = fileWriterChooser
        self._scanPointList: list[ScanPoint] = list()

    def __getitem__(self, index: int) -> ScanPoint:
        return self._scanPointList[index]  # FIXME apply ScanTransformation

    def __len__(self) -> int:
        return len(self._scanPointList)

    def setScanPoints(self, scanPointIterable: Iterable[ScanPoint]) -> None:
        self._scanPointList = [scanPoint for scanPoint in scanPointIterable]
        self.notifyObservers()

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveScan(self, filePath: Path, fileFilter: str) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, SimpleScanDictionary.createFromUnnamedSequence(self))


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

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializer.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._initializer.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.openScan(filePath, fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
        return self._initializer.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._initializer.getSaveFileFilter()

    def saveScan(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.saveScan(filePath, fileFilter)

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
