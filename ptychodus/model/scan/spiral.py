from __future__ import annotations
from decimal import Decimal

import numpy

from ...api.scan import ScanPoint
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings


class SpiralScanInitializer(ScanInitializer):

    def __init__(self, parameters: ScanInitializerParameters, numberOfPoints: int,
                 radiusScalarInMeters: Decimal, angularStepInTurns: Decimal) -> None:
        super().__init__(parameters)
        self._numberOfPoints = numberOfPoints
        self._radiusScalarInMeters = radiusScalarInMeters
        self._angularStepInTurns = angularStepInTurns

    @classmethod
    def createFromSettings(cls, parameters: ScanInitializerParameters,
                           settings: ScanSettings) -> SpiralScanInitializer:
        numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        radiusScalarInMeters = settings.radiusScalarInMeters.value
        angularStepInTurns = settings.angularStepXInTurns.value
        return cls(parameters, numberOfPoints, radiusScalarInMeters, angularStepInTurns)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.numberOfPointsX.value = self._numberOfPoints
        settings.numberOfPointsY.value = 1
        settings.radiusScalarInMeters.value = self._radiusScalarInMeters
        settings.angularStepXInTurns.value = self._angularStepInTurns
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

    def getNumberOfPoints(self) -> int:
        return self._numberOfPoints

    def setNumberOfPoints(self, numberOfPoints: int) -> None:
        if self._numberOfPoints != numberOfPoints:
            self._numberOfPoints = numberOfPoints
            self.notifyObservers()

    def getRadiusScalarInMeters(self) -> Decimal:
        return self._radiusScalarInMeters

    def setRadiusScalarInMeters(self, radiusScalarInMeters: Decimal) -> None:
        if self._radiusScalarInMeters != radiusScalarInMeters:
            self._radiusScalarInMeters = radiusScalarInMeters
            self.notifyObservers()

    def getAngularStepInTurns(self) -> Decimal:
        return self._angularStepInTurns

    def setAngularStepInTurns(self, angularStepInTurns: Decimal) -> None:
        if self._angularStepInTurns != angularStepInTurns:
            self._angularStepInTurns = angularStepInTurns
            self.notifyObservers()

    def _getPoint(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        radius = self._radiusScalarInMeters * Decimal(index).sqrt()
        theta = 2 * numpy.pi * index * float(self._angularStepInTurns)

        return ScanPoint(
            radius * Decimal(numpy.cos(theta)),
            radius * Decimal(numpy.sin(theta)),
        )

    def __len__(self) -> int:
        return self._numberOfPoints
