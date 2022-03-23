from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Optional
import csv
import logging

import numpy

from .geometry import Interval, Box
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class ScanSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Snake')
        self.customFilePath = settingsGroup.createPathEntry('CustomFilePath', None)
        self.extentX = settingsGroup.createIntegerEntry('ExtentX', 10)
        self.extentY = settingsGroup.createIntegerEntry('ExtentY', 10)
        self.stepSizeXInMeters = settingsGroup.createRealEntry('StepSizeXInMeters', '1e-6')
        self.stepSizeYInMeters = settingsGroup.createRealEntry('StepSizeYInMeters', '1e-6')
        self.jitterRadiusInPixels = settingsGroup.createRealEntry('JitterRadiusInPixels', '0')
        self.transform = settingsGroup.createStringEntry('Transform', '+X+Y')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ScanSettings:
        settings = cls(settingsRegistry.createGroup('Scan'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


@dataclass(frozen=True)
class ScanPoint:
    x: Decimal
    y: Decimal


class CartesianScanInitializer(Sequence[ScanPoint]):
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

        x *= self._settings.stepSizeXInMeters.value
        y *= self._settings.stepSizeYInMeters.value

        return ScanPoint(x, y)

    def __len__(self) -> int:
        nx = self._settings.extentX.value
        ny = self._settings.extentY.value
        return nx * ny

    def __str__(self) -> str:
        return 'Snake' if self._snake else 'Raster'


class SpiralScanInitializer(Sequence[ScanPoint]):
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

    def __str__(self) -> str:
        return 'Spiral'


class ScanPointParseError(Exception):
    pass


class CustomScanInitializer(Sequence[ScanPoint]):
    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings
        self._scanPointList: list[ScanPoint] = list()

    def setScanPoints(self, scanPointIterable: Iterable[ScanPoint]) -> None:
        self._scanPointList = [point for point in scanPointIterable]

    def __getitem__(self, index: int) -> ScanPoint:
        return self._scanPointList[index]

    def __len__(self) -> int:
        return len(self._scanPointList)

    def __str__(self) -> str:
        return 'Custom'


class ScanPointTransform(Enum):
    PXPY = 0x0
    MXPY = 0x1
    PXMY = 0x2
    MXMY = 0x3
    PYPX = 0x4
    PYMX = 0x5
    MYPX = 0x6
    MYMX = 0x7

    @property
    def negateX(self) -> bool:
        return self.value & 1 != 0

    @property
    def negateY(self) -> bool:
        return self.value & 2 != 0

    @property
    def swapXY(self) -> bool:
        return self.value & 4 != 0

    def isNameMatchFor(self, query: str) -> bool:
        queryCaseFolded = query.casefold()

        isMatch = False
        isMatch |= (queryCaseFolded == self.name.casefold())
        isMatch |= (queryCaseFolded == repr(self))
        isMatch |= (queryCaseFolded == str(self))
        return isMatch

    def __call__(self, point: ScanPoint) -> ScanPoint:
        xp = -point.x if self.negateX else point.x
        yp = -point.y if self.negateY else point.y
        return ScanPoint(yp, xp) if self.swapXY else ScanPoint(xp, yp)

    def __str__(self) -> str:
        xp = '\u2212x' if self.negateX else '\u002Bx'
        yp = '\u2212y' if self.negateY else '\u002By'
        return f'({yp}, {xp})' if self.swapXY else f'({xp}, {yp})'

    def __repr__(self) -> str:
        xp = '-x' if self.negateX else '+x'
        yp = '-y' if self.negateY else '+y'
        return f'{yp}{xp}' if self.swapXY else f'{xp}{yp}'


class Scan(Sequence[ScanPoint], Observable, Observer):
    FILE_FILTER = 'Comma-Separated Values Files (*.csv)'

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings
        self._scanPointList: list[ScanPoint] = list()
        self._boundingBoxInMeters: Optional[Box[Decimal]] = None
        self._transform = ScanPointTransform.PXPY

    @classmethod
    def createInstance(cls, settings: ScanSettings) -> Scan:
        scan = cls(settings)
        scan.setTransformFromSettings()
        settings.transform.addObserver(scan)
        return scan

    def __getitem__(self, index: int) -> ScanPoint:
        return self._transform(self._scanPointList[index])

    def __len__(self) -> int:
        return len(self._scanPointList)

    def getTransform(self) -> str:
        return str(self._transform)

    def setTransform(self, transform: ScanPointTransform) -> None:
        if transform != self._transform:
            self._transform = transform
            self._settings.transform.value = repr(self._transform)
            self._updateBoundingBox()
            self.notifyObservers()

    def setTransformByName(self, name: str) -> None:
        try:
            transform = next(xform for xform in ScanPointTransform if xform.isNameMatchFor(name))
        except StopIteration:
            logger.debug(f'Invalid transform \"{name}\"')
            return

        self.setTransform(transform)

    def setTransformFromSettings(self) -> None:
        self.setTransformByName(self._settings.transform.value)

    def setScanPoints(self, scanPointIterable: Iterable[ScanPoint]) -> None:
        self._scanPointList = [scanPoint for scanPoint in scanPointIterable]
        self._updateBoundingBox()
        self.notifyObservers()

    def read(self, filePath: Path) -> list[ScanPoint]:
        scanPointList = list()

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) < 2:
                    raise ScanPointParseError()

                x = Decimal(row[1])
                y = Decimal(row[0])
                point = ScanPoint(x, y)

                scanPointList.append(point)

        self._settings.customFilePath.value = filePath
        self.setScanPoints(scanPointList)

    def write(self, filePath: Path) -> None:
        with open(filePath, 'wt') as csvFile:
            for point in self._scanPointList:
                csvFile.write(f'{point.y},{point.x}\n')

    def getBoundingBoxInMeters(self) -> Optional[Box[Decimal]]:
        return self._boundingBoxInMeters

    def _updateBoundingBox(self) -> None: # FIXME
        boundingBoxInMeters = None
        scanPointIterator = iter(self)

        try:
            scanPoint = next(scanPointIterator)
            xint = Interval(scanPoint.x, scanPoint.x)
            yint = Interval(scanPoint.y, scanPoint.y)
            boundingBoxInMeters = Box[Decimal]((xint, yint))

            while True:
                scanPoint = next(scanPointIterator)
                boundingBoxInMeters[0].hull(scanPoint.x)
                boundingBoxInMeters[1].hull(scanPoint.y)
        except StopIteration:
            pass

        self._boundingBoxInMeters = boundingBoxInMeters

    def update(self, observable: Observable) -> None:
        if observable is self._settings.transform:
            self.setTransformFromSettings()


class ScanInitializer(Observable, Observer):
    def __init__(self, settings: ScanSettings, scan: Scan, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan
        self._reinitObservable = reinitObservable
        self._customInitializer = CustomScanInitializer(settings)
        self._initializer = self._customInitializer
        self._initializerList: list[Sequence[ScanPoint]] = [self._customInitializer]

    @classmethod
    def createInstance(cls, settings: ScanSettings, scan: Scan,
                       reinitObservable: Observable) -> ScanInitializer:
        initializer = cls(settings, scan, reinitObservable)
        initializer.addInitializer(CartesianScanInitializer.createRasterInstance(settings))
        initializer.addInitializer(CartesianScanInitializer.createSnakeInstance(settings))
        initializer.addInitializer(SpiralScanInitializer(settings))
        initializer.setInitializerFromSettings()
        settings.initializer.addObserver(initializer)
        reinitObservable.addObserver(initializer)
        return initializer

    def addInitializer(self, initializer: Sequence[ScanPoint]) -> None:
        self._initializerList.append(initializer)

    def getInitializerList(self) -> list[str]:
        return [str(initializer) for initializer in self._initializerList]

    def getInitializer(self) -> str:
        return str(self._initializer)

    def _setInitializer(self, initializer: Sequence[ScanPoint]) -> None:
        if initializer is not self._initializer:
            self._initializer = initializer
            self._settings.initializer.value = str(self._initializer)
            self.notifyObservers()

    def setInitializer(self, name: str) -> None:
        try:
            initializer = next(ini for ini in self._initializerList
                               if name.casefold() == str(ini).casefold())
        except StopIteration:
            logger.debug(f'Invalid initializer \"{name}\"')
            return

        self._setInitializer(initializer)

    def setInitializerFromSettings(self) -> None:
        self.setInitializer(self._settings.initializer.value)

    def initializeScan(self) -> None:
        self._scan.setScanPoints(self._initializer)

    def openScan(self, filePath: Path) -> None:
        self._settings.customFilePath.value = filePath
        self._scan.read(filePath)
        self._customInitializer.setScanPoints(self._scan)
        self._setInitializer(self._customInitializer)

    def saveScan(self, filePath: Path) -> None:
        self._scan.write(filePath)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setInitializerFromSettings()
        elif observable is self._reinitObservable:
            self.initializeScan()


class ScanPresenter(Observable, Observer):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: ScanSettings, scan: Scan, scanInitializer: ScanInitializer) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan
        self._scanInitializer = scanInitializer

    @classmethod
    def createInstance(cls, settings: ScanSettings, scan: Scan, scanInitializer: ScanInitializer) -> ScanPresenter:
        presenter = cls(settings, scan, scanInitializer)
        settings.addObserver(presenter)
        scan.addObserver(presenter)
        scanInitializer.addObserver(presenter)
        return presenter

    def openScan(self, filePath: Path) -> None:
        self._scanInitializer.openScan(filePath)

    def saveScan(self, filePath: Path) -> None:
        self._scanInitializer.saveScan(filePath)

    def getInitializerList(self) -> list[str]:
        return self._scanInitializer.getInitializerList()

    def getInitializer(self) -> str:
        return self._scanInitializer.getInitializer()

    def setInitializer(self, name: str) -> None:
        self._scanInitializer.setInitializer(name)

    def initializeScan(self) -> None:
        self._scanInitializer.initializeScan()

    def getTransformList(self) -> list[str]:
        return [str(xform) for xform in ScanPointTransform]

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

    def getJitterRadiusInPixels(self) -> Decimal:
        return self._settings.jitterRadiusInPixels.value

    def setJitterRadiusInPixels(self, value: Decimal) -> None:
        self._settings.jitterRadiusInPixels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._scanInitializer:
            self.notifyObservers()
