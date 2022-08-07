from __future__ import annotations
from decimal import Decimal

import numpy

from ...api.scan import ScanPoint
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings


class SpiralScanInitializer(ScanInitializer):

    def __init__(self, parameters: ScanInitializerParameters, stepSizeXInMeters: Decimal,
                 stepSizeYInMeters: Decimal, numberOfPoints: int) -> None:
        super().__init__(parameters)
        self._stepSizeXInMeters = stepSizeXInMeters
        self._stepSizeYInMeters = stepSizeYInMeters
        self._numberOfPoints = numberOfPoints

    @classmethod
    def createFromSettings(cls, parameters: ScanInitializerParameters,
                           settings: ScanSettings) -> SpiralScanInitializer:
        stepSizeXInMeters = settings.stepSizeXInMeters.value
        stepSizeYInMeters = settings.stepSizeYInMeters.value
        numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        return cls(parameters, stepSizeXInMeters, stepSizeYInMeters, numberOfPoints)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.initializer.value = self.variant
        settings.stepSizeXInMeters.value = self._stepSizeXInMeters
        settings.stepSizeYInMeters.value = self._stepSizeYInMeters
        settings.numberOfPointsX.value = self._numberOfPoints
        settings.numberOfPointsY.value = 1
        super().syncToSettings(settings)

    @property
    def category(self) -> str:
        return 'Spiral'

    @property
    def variant(self) -> str:
        return 'Archimedean'

    def getStepSizeXInMeters(self) -> Decimal:
        return self._stepSizeXInMeters

    def setStepSizeXInMeters(self, stepSizeXInMeters: Decimal) -> None:
        if self._stepSizeXInMeters != stepSizeXInMeters:
            self._stepSizeXInMeters = stepSizeXInMeters
            self.notifyObservers()

    def getStepSizeYInMeters(self) -> Decimal:
        return self._stepSizeYInMeters

    def setStepSizeYInMeters(self, stepSizeYInMeters: Decimal) -> None:
        if self._stepSizeYInMeters != stepSizeYInMeters:
            self._stepSizeYInMeters = stepSizeYInMeters
            self.notifyObservers()

    def getNumberOfPoints(self) -> int:
        return self._numberOfPoints

    def setNumberOfPoints(self, numberOfPoints: int) -> None:
        if self._numberOfPoints != numberOfPoints:
            self._numberOfPoints = numberOfPoints
            self.notifyObservers()

    def _getPoint(self, index: int) -> ScanPoint:
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

        x = sqrtIndex * cosTheta * self._stepSizeXInMeters
        y = sqrtIndex * sinTheta * self._stepSizeYInMeters

        return ScanPoint(x, y)

    def __len__(self) -> int:
        return self._numberOfPoints
