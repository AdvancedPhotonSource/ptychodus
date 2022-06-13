from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Optional
import csv
import logging

import numpy

from ..api.geometry import Interval, Box
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry
from ..api.scan import ScanPoint, ScanPointSequence, ScanPointTransform, ScanFileReader, ScanFileWriter
from ..api.settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class ScanSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Snake')
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


class CartesianScanInitializer(ScanPointSequence):

    def __init__(self, settings: ScanSettings, snake: bool) -> None:
        super().__init__()
        self._settings = settings
        self._snake = snake

    @classmethod
    def createRasterInstance(cls, settings: ScanSettings) -> CartesianScanInitializer:
        return cls(settings, False)

    @classmethod
    def createSnakeInstance(cls, settings: ScanSettings) -> CartesianScanInitializer:
        return cls(settings, True)

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        nx = self._settings.extentX.value
        y, x = divmod(index, nx)

        if self._snake and y & 1:
            x = nx - 1 - x

        xf = x * self._settings.stepSizeXInMeters.value
        yf = y * self._settings.stepSizeYInMeters.value

        return ScanPoint(xf, yf)

    def __len__(self) -> int:
        nx = self._settings.extentX.value
        ny = self._settings.extentY.value
        return nx * ny


class SpiralScanInitializer(ScanPointSequence):

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        # theta = omega * t
        # r = a + b * theta
        # x = r * numpy.cos(theta)
        # y = r * numpy.sin(theta)

        sqrtIndex = Decimal(index).sqrt()

        # TODO generalize parameters and redo without casting to float
        theta = float(4 * sqrtIndex)
        cosTheta = Decimal(numpy.cos(theta))
        sinTheta = Decimal(numpy.sin(theta))

        x = sqrtIndex * cosTheta * self._settings.stepSizeXInMeters.value
        y = sqrtIndex * sinTheta * self._settings.stepSizeYInMeters.value

        return ScanPoint(x, y)

    def __len__(self) -> int:
        nx = self._settings.extentX.value
        ny = self._settings.extentY.value
        return nx * ny


class JitteredScanInitializer(ScanPointSequence):

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 scanPointSequence: ScanPointSequence) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._scanPointSequence = scanPointSequence

    def __getitem__(self, index: int) -> ScanPoint:
        scanPoint = self._scanPointSequence[index]

        if self._settings.jitterRadiusInMeters.value > 0:
            rad = Decimal(self._rng.uniform())
            dirX = Decimal(self._rng.normal())
            dirY = Decimal(self._rng.normal())

            scalar = self._settings.jitterRadiusInMeters.value \
                    * (rad / (dirX ** 2 + dirY ** 2)).sqrt()
            scanPoint = ScanPoint(scanPoint.x + scalar * dirX, scanPoint.y + scalar * dirY)

        return scanPoint

    def __len__(self) -> int:
        return len(self._scanPointSequence)


class FileScanInitializer(ScanPointSequence, Observer):

    def __init__(self, settings: ScanSettings,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser
        self._pointList: list[ScanPoint] = list()

    @classmethod
    def createInstance(cls, settings: ScanSettings,
                       fileReaderChooser: PluginChooser[ScanFileReader]) -> FileScanInitializer:
        initializer = cls(settings, fileReaderChooser)

        settings.inputFileType.addObserver(initializer)
        initializer._fileReaderChooser.addObserver(initializer)
        initializer._syncFileReaderFromSettings()

        settings.inputFilePath.addObserver(initializer)
        initializer._openScanFromSettings()

        return initializer

    def __getitem__(self, index: int) -> ScanPoint:
        return self._pointList[index]

    def __len__(self) -> int:
        return len(self._pointList)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openScan(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            pointIterable = fileReader.read(filePath)
            self._pointList = [point for point in pointIterable]

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.inputFilePath.value == filePath:
            self._openScan(filePath)

        self._settings.inputFilePath.value = filePath

    def _openScanFromSettings(self) -> None:
        self._openScan(self._settings.inputFilePath.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.inputFileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.inputFilePath:
            self._openScanFromSettings()


class Scan(ScanPointSequence, Observable, Observer):

    @staticmethod
    def _createTransformEntry(xform: ScanPointTransform) -> PluginEntry[ScanPointTransform]:
        return PluginEntry[ScanPointTransform](simpleName=xform.simpleName,
                                               displayName=xform.displayName,
                                               strategy=xform)

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings
        self._scanPointList: list[ScanPoint] = list()
        self._boundingBoxInMeters: Optional[Box[Decimal]] = None
        self._transformChooser = PluginChooser[ScanPointTransform].createFromList(
            [Scan._createTransformEntry(xform) for xform in ScanPointTransform])

    @classmethod
    def createInstance(cls, settings: ScanSettings) -> Scan:
        scan = cls(settings)
        settings.transform.addObserver(scan)
        scan._transformChooser.addObserver(scan)
        scan._syncTransformFromSettings()
        return scan

    def __getitem__(self, index: int) -> ScanPoint:
        transform = self._transformChooser.getCurrentStrategy()
        return transform(self._scanPointList[index])

    def __len__(self) -> int:
        return len(self._scanPointList)

    def setScanPoints(self, scanPointIterable: Iterable[ScanPoint]) -> None:
        self._scanPointList = [scanPoint for scanPoint in scanPointIterable]
        self._updateBoundingBox()
        self.notifyObservers()

    def getBoundingBoxInMeters(self) -> Optional[Box[Decimal]]:
        return self._boundingBoxInMeters

    def _updateBoundingBox(self) -> None:
        boundingBoxInMeters = None
        scanPointIterator = iter(self)

        try:
            scanPoint = next(scanPointIterator)
            xint = Interval[Decimal](scanPoint.x, scanPoint.x)
            yint = Interval[Decimal](scanPoint.y, scanPoint.y)
            boundingBoxInMeters = Box[Decimal]((xint, yint))

            while True:
                scanPoint = next(scanPointIterator)
                boundingBoxInMeters[0].hull(scanPoint.x)
                boundingBoxInMeters[1].hull(scanPoint.y)
        except StopIteration:
            pass

        self._boundingBoxInMeters = boundingBoxInMeters

    def getTransformList(self) -> list[str]:
        return self._transformChooser.getDisplayNameList()

    def getTransform(self) -> str:
        return self._transformChooser.getCurrentDisplayName()

    def setTransform(self, name: str) -> None:
        self._transformChooser.setFromDisplayName(name)

    def _syncTransformFromSettings(self) -> None:
        self._transformChooser.setFromSimpleName(self._settings.transform.value)

    def _syncTransformToSettings(self) -> None:
        self._updateBoundingBox()
        self._settings.transform.value = self._transformChooser.getCurrentSimpleName()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.transform:
            self._syncTransformFromSettings()
        elif observable is self._transformChooser:
            self._syncTransformToSettings()


class ScanInitializer(Observable, Observer):

    def __init__(self, settings: ScanSettings, scan: Scan, fileInitializer: FileScanInitializer,
                 fileWriterChooser: PluginChooser[ScanFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable
        self._fileInitializer = fileInitializer
        self._initializerChooser = PluginChooser[ScanPointSequence](PluginEntry[ScanPointSequence](
            simpleName='FromFile', displayName='From File', strategy=self._fileInitializer))

    @classmethod
    def createInstance(cls, rng: numpy.random.Generator, settings: ScanSettings, scan: Scan,
                       fileInitializer: FileScanInitializer,
                       fileWriterChooser: PluginChooser[ScanFileWriter],
                       reinitObservable: Observable) -> ScanInitializer:
        initializer = cls(settings, scan, fileInitializer, fileWriterChooser, reinitObservable)

        spiralInit = PluginEntry[ScanPointSequence](simpleName='Spiral',
                                                    displayName='Spiral',
                                                    strategy=JitteredScanInitializer(
                                                        rng, settings,
                                                        SpiralScanInitializer(settings)))
        initializer._initializerChooser.addStrategy(spiralInit)

        snakeInit = PluginEntry[ScanPointSequence](
            simpleName='Snake',
            displayName='Snake',
            strategy=JitteredScanInitializer(
                rng, settings, CartesianScanInitializer.createSnakeInstance(settings)))
        initializer._initializerChooser.addStrategy(snakeInit)

        rasterInit = PluginEntry[ScanPointSequence](
            simpleName='Raster',
            displayName='Raster',
            strategy=JitteredScanInitializer(
                rng, settings, CartesianScanInitializer.createRasterInstance(settings)))
        initializer._initializerChooser.addStrategy(rasterInit)

        settings.initializer.addObserver(initializer)
        initializer._initializerChooser.addObserver(initializer)
        initializer._syncInitializerFromSettings()
        reinitObservable.addObserver(initializer)

        return initializer

    def getInitializerList(self) -> list[str]:
        return self._initializerChooser.getDisplayNameList()

    def getInitializer(self) -> str:
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializer(self, name: str) -> None:
        self._initializerChooser.setFromDisplayName(name)

    def initializeScan(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Scan')
        self._scan.setScanPoints(initializer)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileInitializer.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._fileInitializer.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._fileInitializer.openScan(filePath, fileFilter)
        self._initializerChooser.setToDefault()
        self.initializeScan()

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveScan(self, filePath: Path, fileFilter: str) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, self._scan)

    def _syncInitializerFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value)

    def _syncInitializerToSettings(self) -> None:
        self._settings.initializer.value = self._initializerChooser.getCurrentSimpleName()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self._syncInitializerFromSettings()
        elif observable is self._initializerChooser:
            self._syncInitializerToSettings()
        elif observable is self._reinitObservable:
            self.initializeScan()


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
