from __future__ import annotations
from decimal import Decimal
from typing import Final

import numpy

from ...api.probe import Probe
from .apparatus import Apparatus
from .repository import ProbeInitializer
from .settings import ProbeSettings
from .sizer import ProbeSizer


class DiskProbeInitializer(ProbeInitializer):
    SIMPLE_NAME: Final[str] = 'Disk'
    DISPLAY_NAME: Final[str] = 'Disk'

    def __init__(self, sizer: ProbeSizer, apparatus: Apparatus) -> None:
        super().__init__()
        self._sizer = sizer
        self._apparatus = apparatus
        self._diameterInMeters = Decimal('1.5e-6')
        self._isTestPattern = False

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self._diameterInMeters = settings.diskDiameterInMeters.value
        self.notifyObservers()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.diskDiameterInMeters.value = self._diameterInMeters

    def __call__(self) -> Probe:
        extent = self._sizer.getImageExtent()
        pixelGeometry = self._apparatus.getObjectPlanePixelGeometry()
        Y, X = numpy.mgrid[:extent.heightInPixels, :extent.widthInPixels]
        X_px = X - (extent.widthInPixels - 1) / 2
        Y_px = Y - (extent.heightInPixels - 1) / 2

        if self._isTestPattern:
            Rmax_px = min(extent.widthInPixels, extent.heightInPixels) / 2
            R_px = numpy.hypot(X_px, Y_px)
            array = numpy.where(R_px < Rmax_px, X_px + 1j * Y_px, 0j)
        else:
            Rmax_m = self._diameterInMeters / 2
            X_m = X_px * float(pixelGeometry.widthInMeters)
            Y_m = Y_px * float(pixelGeometry.heightInMeters)
            R_m = numpy.hypot(X_m, Y_m)

            array = numpy.where(R_m < Rmax_m, 1 + 0j, 0j)
            array /= numpy.sqrt(numpy.sum(numpy.abs(array)**2))

        return Probe(array)

    def getDiameterInMeters(self) -> Decimal:
        return self._diameterInMeters

    def setDiameterInMeters(self, value: Decimal) -> None:
        if self._diameterInMeters != value:
            self._diameterInMeters = value
            self.notifyObservers()

    def isTestPattern(self) -> bool:
        return self._isTestPattern

    def setTestPattern(self, value: bool) -> None:
        if self._isTestPattern != value:
            self._isTestPattern = value
            self.notifyObservers()
