from collections.abc import Iterator
from decimal import Decimal
from typing import Final

import numpy

from ...api.scan import Scan, ScanPoint
from .repository import ScanInitializer
from .settings import ScanSettings

__all__ = [
    'LissajousScanInitializer',
]


class LissajousScan(Scan):

    def __init__(self) -> None:
        super().__init__()
        self.numberOfPoints = 100
        self.amplitudeXInMeters = Decimal('4.5e-6')
        self.amplitudeYInMeters = Decimal('4.5e-6')
        self.angularStepXInTurns = Decimal('0.03')
        self.angularStepYInTurns = Decimal('0.04')
        self.angularShiftInTurns = Decimal('0.25')

    @property
    def nameHint(self) -> str:
        return 'Lissajous'

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self)):
            yield index

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        twoPi = 2 * numpy.pi
        thetaX = twoPi * float(self.angularStepXInTurns * index + self.angularShiftInTurns)
        thetaY = twoPi * float(self.angularStepYInTurns * index)

        return ScanPoint(
            self.amplitudeXInMeters * Decimal(numpy.sin(thetaX)),
            self.amplitudeYInMeters * Decimal(numpy.sin(thetaY)),
        )

    def __len__(self) -> int:
        return self.numberOfPoints


class LissajousScanInitializer(ScanInitializer):
    NAME: Final[str] = 'Lissajous'

    def __init__(self) -> None:
        super().__init__()
        self._scan = LissajousScan()

    @property
    def simpleName(self) -> str:
        return self._scan.nameHint

    @property
    def displayName(self) -> str:
        return self.NAME

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._scan.numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        self._scan.amplitudeXInMeters = settings.amplitudeXInMeters.value
        self._scan.amplitudeYInMeters = settings.amplitudeYInMeters.value
        self._scan.angularStepXInTurns = settings.angularStepXInTurns.value
        self._scan.angularStepYInTurns = settings.angularStepYInTurns.value
        self._scan.angularShiftInTurns = settings.angularShiftInTurns.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.numberOfPointsX.value = self._scan.numberOfPoints
        settings.numberOfPointsY.value = 1
        settings.amplitudeXInMeters.value = self._scan.amplitudeXInMeters
        settings.amplitudeYInMeters.value = self._scan.amplitudeYInMeters
        settings.angularStepXInTurns.value = self._scan.angularStepXInTurns
        settings.angularStepYInTurns.value = self._scan.angularStepYInTurns
        settings.angularShiftInTurns.value = self._scan.angularShiftInTurns

    def __call__(self) -> Scan:
        return self._scan

    def getNumberOfPoints(self) -> int:
        return self._scan.numberOfPoints

    def setNumberOfPoints(self, numberOfPoints: int) -> None:
        if self._scan.numberOfPoints != numberOfPoints:
            self._scan.numberOfPoints = numberOfPoints
            self.notifyObservers()

    def getAmplitudeXInMeters(self) -> Decimal:
        return self._scan.amplitudeXInMeters

    def setAmplitudeXInMeters(self, amplitudeXInMeters: Decimal) -> None:
        if self._scan.amplitudeXInMeters != amplitudeXInMeters:
            self._scan.amplitudeXInMeters = amplitudeXInMeters
            self.notifyObservers()

    def getAmplitudeYInMeters(self) -> Decimal:
        return self._scan.amplitudeYInMeters

    def setAmplitudeYInMeters(self, amplitudeYInMeters: Decimal) -> None:
        if self._scan.amplitudeYInMeters != amplitudeYInMeters:
            self._scan.amplitudeYInMeters = amplitudeYInMeters
            self.notifyObservers()

    def getAngularStepXInTurns(self) -> Decimal:
        return self._scan.angularStepXInTurns

    def setAngularStepXInTurns(self, angularStepXInTurns: Decimal) -> None:
        if self._scan.angularStepXInTurns != angularStepXInTurns:
            self._scan.angularStepXInTurns = angularStepXInTurns
            self.notifyObservers()

    def getAngularStepYInTurns(self) -> Decimal:
        return self._scan.angularStepYInTurns

    def setAngularStepYInTurns(self, angularStepYInTurns: Decimal) -> None:
        if self._scan.angularStepYInTurns != angularStepYInTurns:
            self._scan.angularStepYInTurns = angularStepYInTurns
            self.notifyObservers()

    def getAngularShiftInTurns(self) -> Decimal:
        return self._scan.angularShiftInTurns

    def setAngularShiftInTurns(self, angularShiftInTurns: Decimal) -> None:
        if self._scan.angularShiftInTurns != angularShiftInTurns:
            self._scan.angularShiftInTurns = angularShiftInTurns
            self.notifyObservers()
