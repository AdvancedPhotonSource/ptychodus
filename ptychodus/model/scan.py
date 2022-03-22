from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple
import csv

import numpy

from .geometry import Interval, Box
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup


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
        self.transformXY = settingsGroup.createStringEntry('TransformXY', '+X+Y')

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


class ScanPointParseError(Exception):
    pass


class ScanPointIO:
    FILE_FILTER = 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> list[ScanPoint]:
        point_list = list()

        with open(filePath, newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                if row[0].startswith('#'):
                    continue

                if len(row) < 2:
                    raise ScanPointParseError()

                x = Decimal(row[1])
                y = Decimal(row[0])
                point = ScanPoint(x, y)

                point_list.append(point)

        return point_list

    def write(self, filePath: Path, scanPointList: list[ScanPoint]) -> None:
        with open(filePath, 'wt') as csv_file:
            for point in scanPointList:
                csv_file.write(f'{point.y},{point.x}\n')


class ScanSequence(Sequence, Observable, Observer):
    pass


class ScanSizer(Observable, Observer):
    def __init__(self, scanSequence: ScanSequence) -> None:
        super().__init__()
        self._scanSequence = scanSequence
        self._boundingBoxInMeters: Optional[Box[Decimal]] = None

    @classmethod
    def createInstance(cls, scanSequence: ScanSequence) -> ScanSizer:
        sizer = cls(scanSequence)
        scanSequence.addObserver(sizer)
        return sizer

    def getBoundingBoxInMeters(self) -> Optional[Box[Decimal]]:
        return self._boundingBoxInMeters

    def _updateBoundingBox(self) -> None:
        pointIter = iter(self._scanSequence)

        try:
            point = next(pointIter)
        except StopIteration:
            self._boundingBoxInMeters = None
            return

        xint = Interval(point.x, point.x)
        yint = Interval(point.y, point.y)

        for point in pointIter:
            xint.hull(point.x)
            yint.hull(point.y)

        self._boundingBoxInMeters = Box[Decimal]((xint, yint))
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._scanSequence:
            self._updateBoundingBox()


class CartesianScanSequence(ScanSequence):
    def __init__(self, settings: ScanSettings, snake: bool) -> None:
        super().__init__()
        self._settings = settings
        self._snake = snake

    @classmethod
    def createRasterInstance(cls, settings: ScanSettings) -> CartesianScanSequence:
        generator = cls(settings, False)
        settings.extentX.addObserver(generator)
        settings.extentY.addObserver(generator)
        settings.stepSizeXInMeters.addObserver(generator)
        settings.stepSizeYInMeters.addObserver(generator)
        return generator

    @classmethod
    def createSnakeInstance(cls, settings: ScanSettings) -> CartesianScanSequence:
        generator = cls(settings, True)
        settings.extentX.addObserver(generator)
        settings.extentY.addObserver(generator)
        settings.stepSizeXInMeters.addObserver(generator)
        settings.stepSizeYInMeters.addObserver(generator)
        return generator

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

    def update(self, observable: Observable) -> None:
        self.notifyObservers()


class SpiralScanSequence(ScanSequence):
    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: ScanSettings) -> SpiralScanSequence:
        generator = cls(settings)
        settings.extentX.addObserver(generator)
        settings.extentY.addObserver(generator)
        settings.stepSizeXInMeters.addObserver(generator)
        settings.stepSizeYInMeters.addObserver(generator)
        return generator

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

    def update(self, observable: Observable) -> None:
        self.notifyObservers()


class CustomScanSequence(ScanSequence):
    def __init__(self) -> None:
        super().__init__()
        self._pointList: list[ScanPoint] = list()

    def __getitem__(self, index: int) -> ScanPoint:
        return self._pointList[index]

    def __len__(self) -> int:
        return len(self._pointList)

    def __str__(self) -> str:
        return 'Custom'

    def setPointList(self, pointList: list[ScanPoint]) -> None:
        self._pointList = pointList
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        pass


class SelectableScanSequence(ScanSequence):
    def __init__(self, settings: ScanSettings, sequenceList: list[ScanSequence]) -> None:
        super().__init__()
        self._settings = settings
        self._sequenceList = sequenceList
        self._sequence = sequenceList[0]

    @classmethod
    def createInstance(cls, settings: ScanSettings) -> SelectableScanSequence:
        sequenceList = list()
        sequenceList.append(CartesianScanSequence.createRasterInstance(settings))
        sequenceList.append(CartesianScanSequence.createSnakeInstance(settings))
        sequenceList.append(SpiralScanSequence.createInstance(settings))
        sequenceList.append(CustomScanSequence())

        selectableSequence = cls(settings, sequenceList)
        selectableSequence.setCurrentScanSequenceFromSettings()
        settings.initializer.addObserver(selectableSequence)

        return selectableSequence

    def getScanSequenceList(self) -> list[str]:
        return [str(sequence) for sequence in self._sequenceList]

    def getCurrentScanSequence(self) -> str:
        return str(self._sequence)

    def setCurrentScanSequence(self, name: str) -> None:
        try:
            sequence = next(seq for seq in self._sequenceList
                            if name.casefold() == str(seq).casefold())
        except StopIteration:
            return

        if sequence is not self._sequence:
            self._sequence.removeObserver(self)
            self._sequence = sequence
            self._settings.initializer.value = str(self._sequence)
            self._sequence.addObserver(self)
            self.notifyObservers()

    def setCurrentScanSequenceFromSettings(self) -> None:
        self.setCurrentScanSequence(self._settings.initializer.value)

    def setCurrentScanSequenceToCustomPointList(self, pointList: list[ScanPoint]) -> None:
        self.setCurrentScanSequence('Custom')
        self._sequence.setPointList(pointList)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._sequence[index]

    def __len__(self) -> int:
        return len(self._sequence)

    def __str__(self) -> str:
        return str(self._sequence)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setCurrentScanSequenceFromSettings()
        if observable is self._sequence:
            self.notifyObservers()


class TransformXY(Enum):
    PXPY = 0x0
    MXPY = 0x1
    PXMY = 0x2
    MXMY = 0x3
    PYPX = 0x4
    PYMX = 0x5
    MYPX = 0x6
    MYMX = 0x7

    def isNameMatchFor(self, query: str) -> bool:
        queryCaseFolded = query.casefold()

        isMatch = False
        isMatch |= (queryCaseFolded == self.name.casefold())
        isMatch |= (queryCaseFolded == repr(self))
        isMatch |= (queryCaseFolded == str(self))
        return isMatch

    def __call__(self, point: ScanPoint) -> ScanPoint:
        xp = -point.x if self.value & 1 else point.x
        yp = -point.y if self.value & 2 else point.y
        return ScanPoint(yp, xp) if self.value & 4 else ScanPoint(xp, yp)

    def __str__(self) -> str:
        xp = '\u2212x' if self.value & 1 else '\u002Bx'
        yp = '\u2212y' if self.value & 2 else '\u002By'
        return f'({yp}, {xp})' if self.value & 4 else f'({xp}, {yp})'

    def __repr__(self) -> str:
        xp = '-x' if self.value & 1 else '+x'
        yp = '-y' if self.value & 2 else '+y'
        return f'{yp}{xp}' if self.value & 4 else f'{xp}{yp}'


class TransformedScanSequence(ScanSequence):
    def __init__(self, settings: ScanSettings, sequence: ScanSequence) -> None:
        super().__init__()
        self._settings = settings
        self._sequence = sequence
        self._transform: Optional[TransformXY] = None

    @classmethod
    def createInstance(cls, settings: ScanSettings,
                       sequence: ScanSequence) -> TransformedScanSequence:
        transformedSequence = cls(settings, sequence)
        transformedSequence.setCurrentTransformXYFromSettings()
        settings.transformXY.addObserver(transformedSequence)
        sequence.addObserver(transformedSequence)
        return transformedSequence

    def getCurrentTransformXY(self) -> str:
        return str(self._transform)

    def setCurrentTransformXY(self, transform: TransformXY) -> None:
        if transform != self._transform:
            self._transform = transform
            self._settings.transformXY.value = repr(self._transform)
            self.notifyObservers()

    def setCurrentTransformXYByName(self, name: str) -> None:
        try:
            transform = next(xform for xform in TransformXY if xform.isNameMatchFor(name))
        except StopIteration:
            return

        self.setCurrentTransformXY(transform)

    def setCurrentTransformXYFromSettings(self) -> None:
        self.setCurrentTransformXYByName(self._settings.transformXY.value)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._transform(self._sequence[index])

    def __len__(self) -> int:
        return len(self._sequence)

    def __str__(self) -> str:
        return str(self._sequence)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.transformXY:
            self.setCurrentTransformXYFromSettings()
        elif observable is self._sequence:
            self.notifyObservers()


class ScanPresenter(Observable, Observer):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: ScanSettings, selectableSequence: SelectableScanSequence,
                 transformedSequence: TransformedScanSequence) -> None:
        super().__init__()
        self._settings = settings
        self._selectableSequence = selectableSequence
        self._transformedSequence = transformedSequence
        self._scanPointIO = ScanPointIO()

    @classmethod
    def createInstance(cls, settings: ScanSettings, selectableSequence: SelectableScanSequence,
                       transformedSequence: TransformedScanSequence) -> ScanPresenter:
        presenter = cls(settings, selectableSequence, transformedSequence)
        transformedSequence.addObserver(presenter)
        return presenter

    def getMinNumberOfScanPoints(self) -> int:
        return 0

    def getMaxNumberOfScanPoints(self) -> int:
        return self.MAX_INT

    def getNumberOfScanPoints(self) -> int:
        return len(self._transformedSequence)

    def getScanPointList(self) -> list[ScanPoint]:
        return [point for point in iter(self._transformedSequence)]

    def openScan(self, filePath: Path) -> None:
        self._settings.customFilePath.value = filePath
        pointList = self._scanPointIO.read(filePath)
        self._selectableSequence.setCurrentScanSequenceToCustomPointList(pointList)

    def saveScan(self, filePath: Path) -> None:
        self._scanPointIO.write(filePath, self.getScanPointList())

    def getScanSequenceList(self) -> list[str]:
        return self._selectableSequence.getScanSequenceList()

    def getCurrentInitializer(self) -> str:
        return self._selectableSequence.getCurrentScanSequence()

    def setCurrentInitializer(self, name: str) -> None:
        self._selectableSequence.setCurrentScanSequence(name)

    def getMinExtentX(self) -> int:
        return 1

    def getMaxExtentX(self) -> int:
        return self.MAX_INT

    def getExtentX(self) -> int:
        return self._clamp(self._settings.extentX.value, self.getMinExtentX(),
                           self.getMaxExtentX())

    def setExtentX(self, value: int) -> None:
        self._settings.extentX.value = value

    def getMinExtentY(self) -> int:
        return 1

    def getMaxExtentY(self) -> int:
        return self.MAX_INT

    def getExtentY(self) -> int:
        return self._clamp(self._settings.extentY.value, self.getMinExtentY(),
                           self.getMaxExtentY())

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

    def getTransformXYList(self) -> list[str]:
        return [str(xform) for xform in TransformXY]

    def getCurrentTransformXY(self) -> str:
        return self._transformedSequence.getCurrentTransformXY()

    def setCurrentTransformXY(self, name: str) -> None:
        self._transformedSequence.setCurrentTransformXYByName(name)

    def initializeScan(self) -> None:
        pass  # FIXME

    @staticmethod
    def _clamp(x, xmin, xmax):
        assert xmin <= xmax
        return max(xmin, min(x, xmax))

    def update(self, observable: Observable) -> None:
        if observable is self._transformedSequence:
            self.notifyObservers()
