from collections.abc import Iterator
from decimal import Decimal
from typing import Final

import numpy

from ...api.scan import ScanPoint
from .repository import ContiguousScanIterator, ScanRepositoryItem
from .settings import ScanSettings


class SpiralScanRepositoryItem(ScanRepositoryItem):
    NAME: Final[str] = 'Spiral'

    def __init__(self) -> None:
        super().__init__()
        self._numberOfPoints = 0
        self._radiusScalarInMeters = Decimal()
        self._angularStepInTurns = Decimal()

    @property
    def name(self) -> str:
        return self.variant

    @property
    def category(self) -> str:
        return self.NAME

    @property
    def variant(self) -> str:
        return 'Fermat'

    @property
    def canActivate(self) -> bool:
        return True

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        self._radiusScalarInMeters = settings.radiusScalarInMeters.value
        self._angularStepInTurns = settings.angularStepXInTurns.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.numberOfPointsX.value = self._numberOfPoints
        settings.numberOfPointsY.value = 1
        settings.radiusScalarInMeters.value = self._radiusScalarInMeters
        settings.angularStepXInTurns.value = self._angularStepInTurns

    def __iter__(self) -> Iterator[int]:
        return ContiguousScanIterator(self)

    def __getitem__(self, index: int) -> ScanPoint:
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
