from __future__ import annotations
from decimal import Decimal
from typing import Final

import numpy

from ...api.probe import Probe
from .apparatus import Apparatus
from .repository import ProbeInitializer
from .settings import ProbeSettings
from .sizer import ProbeSizer


class SuperGaussianProbeInitializer(ProbeInitializer):
    SIMPLE_NAME: Final[str] = 'SuperGaussian'
    DISPLAY_NAME: Final[str] = 'Super Gaussian'

    def __init__(self, sizer: ProbeSizer, apparatus: Apparatus) -> None:
        super().__init__()
        self._sizer = sizer
        self._apparatus = apparatus
        self._annularRadiusInMeters = Decimal()
        self._probeWidthInMeters = Decimal('1.5e-6')
        self._orderParameter = Decimal(1)

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self._annularRadiusInMeters = settings.sgAnnularRadiusInMeters.value
        self._probeWidthInMeters = settings.sgProbeWidthInMeters.value
        self._orderParameter = settings.sgOrderParameter.value
        self.notifyObservers()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.sgAnnularRadiusInMeters.value = self._annularRadiusInMeters
        settings.sgProbeWidthInMeters.value = self._probeWidthInMeters
        settings.sgOrderParameter.value = self._orderParameter

    def __call__(self) -> Probe:
        extent = self._sizer.getImageExtent()
        pixelGeometry = self._apparatus.getObjectPlanePixelGeometry()
        Y, X = numpy.mgrid[:extent.heightInPixels, :extent.widthInPixels]
        X_px = X - (extent.widthInPixels - 1) / 2
        Y_px = Y - (extent.heightInPixels - 1) / 2

        X_m = X_px * float(pixelGeometry.widthInMeters)
        Y_m = Y_px * float(pixelGeometry.heightInMeters)
        R_m = numpy.hypot(X_m, Y_m)

        Z = (R_m - float(self._annularRadiusInMeters)) / float(self._probeWidthInMeters)
        ZP = numpy.power(2 * Z, 2 * float(self._orderParameter))

        array = numpy.exp(-numpy.log(2) * ZP) + 0j
        array /= numpy.sqrt(numpy.sum(numpy.abs(array)**2))

        return Probe(array)

    def getAnnularRadiusInMeters(self) -> Decimal:
        return self._annularRadiusInMeters

    def setAnnularRadiusInMeters(self, value: Decimal) -> None:
        if self._annularRadiusInMeters != value:
            self._annularRadiusInMeters = value
            self.notifyObservers()

    def getFullWidthAtHalfMaximumInMeters(self) -> Decimal:
        return self._probeWidthInMeters

    def setFullWidthAtHalfMaximumInMeters(self, value: Decimal) -> None:
        if self._probeWidthInMeters != value:
            self._probeWidthInMeters = value
            self.notifyObservers()

    def getOrderParameter(self) -> Decimal:
        return max(self._orderParameter, Decimal(1))

    def setOrderParameter(self, value: Decimal) -> None:
        if self._orderParameter != value:
            self._orderParameter = value
            self.notifyObservers()
