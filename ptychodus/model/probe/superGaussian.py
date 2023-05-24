from __future__ import annotations
from decimal import Decimal
from typing import Final

import numpy

from ...api.probe import ProbeArrayType
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
        self._fwhmInMeters = Decimal('1.5e-6')
        self._orderParameter = Decimal(1)

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self._annularRadiusInMeters = settings.sgAnnularRadiusInMeters.value
        self._fwhmInMeters = settings.sgFWHMInMeters.value
        self._orderParameter = settings.sgOrderParameter.value
        self.notifyObservers()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.sgAnnularRadiusInMeters.value = self._annularRadiusInMeters
        settings.sgFWHMInMeters.value = self._fwhmInMeters
        settings.sgOrderParameter.value = self._orderParameter

    def __call__(self) -> ProbeArrayType:
        extent = self._sizer.getProbeExtent()
        cellCentersX = numpy.arange(extent.width) - (extent.width - 1) / 2
        cellCentersY = numpy.arange(extent.height) - (extent.height - 1) / 2
        Y_px, X_px = numpy.meshgrid(cellCentersY, cellCentersX)

        X_m = X_px * float(self._apparatus.getObjectPlanePixelSizeXInMeters())
        Y_m = Y_px * float(self._apparatus.getObjectPlanePixelSizeYInMeters())
        R_m = numpy.hypot(X_m, Y_m)

        Z = (R_m - float(self._annularRadiusInMeters)) / float(self._fwhmInMeters)
        ZP = numpy.power(2 * Z, 2 * float(self._orderParameter))

        return numpy.exp(-numpy.log(2) * ZP) + 0j

    def getAnnularRadiusInMeters(self) -> Decimal:
        return self._annularRadiusInMeters

    def setAnnularRadiusInMeters(self, value: Decimal) -> None:
        if self._annularRadiusInMeters != value:
            self._annularRadiusInMeters = value
            self.notifyObservers()

    def getFullWidthAtHalfMaximumInMeters(self) -> Decimal:
        return self._fwhmInMeters

    def setFullWidthAtHalfMaximumInMeters(self, value: Decimal) -> None:
        if self._fwhmInMeters != value:
            self._fwhmInMeters = value
            self.notifyObservers()

    def getOrderParameter(self) -> Decimal:
        return max(self._orderParameter, Decimal(1))

    def setOrderParameter(self, value: Decimal) -> None:
        if self._orderParameter != value:
            self._orderParameter = value
            self.notifyObservers()
