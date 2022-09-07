from __future__ import annotations
from decimal import Decimal

import numpy

from ...api.scan import ScanPoint
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings


class LissajousScanInitializer(ScanInitializer):

    def __init__(self, parameters: ScanInitializerParameters, numberOfPoints: int,
                 amplitudeXInMeters: Decimal, amplitudeYInMeters: Decimal,
                 angularStepXInTurns: Decimal, angularStepYInTurns: Decimal,
                 angularShiftInTurns: Decimal) -> None:
        super().__init__(parameters)
        self._numberOfPoints = numberOfPoints
        self._amplitudeXInMeters = amplitudeXInMeters
        self._amplitudeYInMeters = amplitudeYInMeters
        self._angularStepXInTurns = angularStepXInTurns
        self._angularStepYInTurns = angularStepYInTurns
        self._angularShiftInTurns = angularShiftInTurns

    @classmethod
    def createFromSettings(cls, parameters: ScanInitializerParameters,
                           settings: ScanSettings) -> LissajousScanInitializer:
        numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        amplitudeXInMeters = settings.amplitudeXInMeters.value
        amplitudeYInMeters = settings.amplitudeYInMeters.value
        angularStepXInTurns = settings.angularStepXInTurns.value
        angularStepYInTurns = settings.angularStepYInTurns.value
        angularShiftInTurns = settings.angularShiftInTurns.value
        return cls(parameters, numberOfPoints, amplitudeXInMeters, amplitudeYInMeters,
                   angularStepXInTurns, angularStepYInTurns, angularShiftInTurns)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.numberOfPointsX.value = self._numberOfPoints
        settings.numberOfPointsY.value = 1
        settings.amplitudeXInMeters.value = self._amplitudeXInMeters
        settings.amplitudeYInMeters.value = self._amplitudeYInMeters
        settings.angularStepXInTurns.value = self._angularStepXInTurns
        settings.angularStepYInTurns.value = self._angularStepYInTurns
        settings.angularShiftInTurns.value = self._angularShiftInTurns
        super().syncToSettings(settings)

    @property
    def nameHint(self) -> str:
        return self.variant

    @property
    def category(self) -> str:
        return 'Curve'

    @property
    def variant(self) -> str:
        return 'Lissajous'

    @property
    def canActivate(self) -> bool:
        return True

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

    def _getPoint(self, index: int) -> ScanPoint:
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
