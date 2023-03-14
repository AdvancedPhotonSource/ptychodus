from collections.abc import Iterator
from decimal import Decimal
from typing import Final

import numpy

from ...api.scan import ScanPoint
from .itemRepository import ScanRepositoryItem
from .settings import ScanSettings


class LissajousScanRepositoryItem(ScanRepositoryItem):
    NAME: Final[str] = 'Lissajous'

    def __init__(self) -> None:
        super().__init__()
        self._numberOfPoints = 0
        self._amplitudeXInMeters = Decimal()
        self._amplitudeYInMeters = Decimal()
        self._angularStepXInTurns = Decimal()
        self._angularStepYInTurns = Decimal()
        self._angularShiftInTurns = Decimal()

    @property
    def name(self) -> str:
        return self.initializer

    @property
    def initializer(self) -> str:
        return self.NAME

    @property
    def canActivate(self) -> bool:
        return True

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        self._amplitudeXInMeters = settings.amplitudeXInMeters.value
        self._amplitudeYInMeters = settings.amplitudeYInMeters.value
        self._angularStepXInTurns = settings.angularStepXInTurns.value
        self._angularStepYInTurns = settings.angularStepYInTurns.value
        self._angularShiftInTurns = settings.angularShiftInTurns.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.numberOfPointsX.value = self._numberOfPoints
        settings.numberOfPointsY.value = 1
        settings.amplitudeXInMeters.value = self._amplitudeXInMeters
        settings.amplitudeYInMeters.value = self._amplitudeYInMeters
        settings.angularStepXInTurns.value = self._angularStepXInTurns
        settings.angularStepYInTurns.value = self._angularStepYInTurns
        settings.angularShiftInTurns.value = self._angularShiftInTurns

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self)):
            yield index

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        twoPi = 2 * numpy.pi
        thetaX = twoPi * float(self._angularStepXInTurns * index + self._angularShiftInTurns)
        thetaY = twoPi * float(self._angularStepYInTurns * index)

        return ScanPoint(
            self._amplitudeXInMeters * Decimal(numpy.sin(thetaX)),
            self._amplitudeYInMeters * Decimal(numpy.sin(thetaY)),
        )

    def __len__(self) -> int:
        return self._numberOfPoints

    def getNumberOfPoints(self) -> int:
        return self._numberOfPoints

    def setNumberOfPoints(self, numberOfPoints: int) -> None:
        if self._numberOfPoints != numberOfPoints:
            self._numberOfPoints = numberOfPoints
            self.notifyObservers()

    def getAmplitudeXInMeters(self) -> Decimal:
        return self._amplitudeXInMeters

    def setAmplitudeXInMeters(self, amplitudeXInMeters: Decimal) -> None:
        if self._amplitudeXInMeters != amplitudeXInMeters:
            self._amplitudeXInMeters = amplitudeXInMeters
            self.notifyObservers()

    def getAmplitudeYInMeters(self) -> Decimal:
        return self._amplitudeYInMeters

    def setAmplitudeYInMeters(self, amplitudeYInMeters: Decimal) -> None:
        if self._amplitudeYInMeters != amplitudeYInMeters:
            self._amplitudeYInMeters = amplitudeYInMeters
            self.notifyObservers()

    def getAngularStepXInTurns(self) -> Decimal:
        return self._angularStepXInTurns

    def setAngularStepXInTurns(self, angularStepXInTurns: Decimal) -> None:
        if self._angularStepXInTurns != angularStepXInTurns:
            self._angularStepXInTurns = angularStepXInTurns
            self.notifyObservers()

    def getAngularStepYInTurns(self) -> Decimal:
        return self._angularStepYInTurns

    def setAngularStepYInTurns(self, angularStepYInTurns: Decimal) -> None:
        if self._angularStepYInTurns != angularStepYInTurns:
            self._angularStepYInTurns = angularStepYInTurns
            self.notifyObservers()

    def getAngularShiftInTurns(self) -> Decimal:
        return self._angularShiftInTurns

    def setAngularShiftInTurns(self, angularShiftInTurns: Decimal) -> None:
        if self._angularShiftInTurns != angularShiftInTurns:
            self._angularShiftInTurns = angularShiftInTurns
            self.notifyObservers()
