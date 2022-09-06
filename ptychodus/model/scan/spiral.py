from __future__ import annotations
from decimal import Decimal

import numpy

from ...api.scan import ScanPoint
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings


class SpiralScanInitializer(ScanInitializer):

    def __init__(self, parameters: ScanInitializerParameters,
            radiusScalarInMeters: Decimal, stepSizeInTurns: Decimal,
            numberOfPoints: int) -> None:
        super().__init__(parameters)
        self._radiusScalarInMeters = radiusScalarInMeters
        self._stepSizeInTurns = stepSizeInTurns
        self._numberOfPoints = numberOfPoints

    @classmethod
    def createFromSettings(cls, parameters: ScanInitializerParameters,
                           settings: ScanSettings) -> SpiralScanInitializer:
        radiusScalarInMeters = settings.radiusScalarInMeters.value
        stepSizeInTurns = settings.stepSizeInTurns.value
        numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        return cls(parameters, radiusScalarInMeters, stepSizeInTurns, numberOfPoints)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.radiusScalarInMeters.value = self._radiusScalarInMeters
        settings.stepSizeInTurns.value = self._stepSizeInTurns
        settings.numberOfPointsX.value = self._numberOfPoints
        settings.numberOfPointsY.value = 1
        super().syncToSettings(settings)

    @property
    def nameHint(self) -> str:
        return self.variant

    @property
    def category(self) -> str:
        return 'Spiral'

    @property
    def variant(self) -> str:
        return 'Fermat'

    def getRadiusScalarInMeters(self) -> Decimal:
        return self._radiusScalarInMeters

    def setRadiusScalarInMeters(self, radiusScalarInMeters: Decimal) -> None:
        if self._radiusScalarInMeters != radiusScalarInMeters:
            self._radiusScalarInMeters = radiusScalarInMeters
            self.notifyObservers()

    def getStepSizeInTurns(self) -> Decimal:
        return self._stepSizeInTurns

    def setStepSizeInTurns(self, stepSizeInTurns: Decimal) -> None:
        if self._stepSizeInTurns != stepSizeInTurns:
            self._stepSizeInTurns = stepSizeInTurns
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

        radius = self._radiusScalarInMeters * Decimal(index).sqrt()
        theta = 2 * numpy.pi * index * float(self._stepSizeInTurns)
        cosTheta = Decimal(numpy.cos(theta))
        sinTheta = Decimal(numpy.sin(theta))

        return ScanPoint(radius * cosTheta, radius * sinTheta)

    def __len__(self) -> int:
        return self._numberOfPoints
