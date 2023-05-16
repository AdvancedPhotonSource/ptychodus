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
        self._probeWidthInMeters = Decimal()
        self._orderParameter = Decimal()

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

    def __call__(self) -> ProbeArrayType:
        extent = self._sizer.getProbeExtent()

        gridCenterX = extent.width / 2
        cellCentersX = (numpy.arange(extent.width) + 0.5 - gridCenterX) / gridCenterX

        gridCenterY = extent.height / 2
        cellCentersY = (numpy.arange(extent.height) + 0.5 - gridCenterY) / gridCenterY

        Y_px, X_px = numpy.meshgrid(cellCentersY, cellCentersX)
        X_m = X_px * float(self._apparatus.getObjectPlanePixelSizeXInMeters())
        Y_m = Y_px * float(self._apparatus.getObjectPlanePixelSizeYInMeters())
        R_m = numpy.hypot(X_m, Y_m)

        Z = (R_m - float(self._annularRadiusInMeters)) / float(self._probeWidthInMeters)
        ZP = numpy.power(Z, 2 * float(self._orderParameter))

        return numpy.exp(-ZP / 2) + 0j

    def getAnnularRadiusInMeters(self) -> Decimal:
        return self._annularRadiusInMeters

    def setAnnularRadiusInMeters(self, value: Decimal) -> None:
        if self._annularRadiusInMeters != value:
            self._annularRadiusInMeters = value
            self.notifyObservers()

    def getProbeWidthInMeters(self) -> Decimal:
        return self._probeWidthInMeters

    def setProbeWidthInMeters(self, value: Decimal) -> None:
        if self._probeWidthInMeters != value:
            self._probeWidthInMeters = value
            self.notifyObservers()

    def getOrderParameter(self) -> Decimal:
        return max(self._orderParameter, Decimal(1))

    def setOrderParameter(self, value: Decimal) -> None:
        if self._orderParameter != value:
            self._orderParameter = value
            self.notifyObservers()
