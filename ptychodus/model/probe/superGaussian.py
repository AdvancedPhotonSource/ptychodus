from __future__ import annotations
from decimal import Decimal

import numpy

from ...api.probe import ProbeArrayType
from ..data import Detector
from .initializer import UnimodalProbeInitializer, UnimodalProbeInitializerParameters
from .settings import ProbeSettings
from .sizer import ProbeSizer


class SuperGaussianProbeInitializer(UnimodalProbeInitializer):

    def __init__(self, parameters: UnimodalProbeInitializerParameters, sizer: ProbeSizer,
                 detector: Detector) -> None:
        super().__init__(parameters)
        self._sizer = sizer
        self._detector = detector
        self._annularRadiusInMeters = Decimal()
        self._probeWidthInMeters = Decimal()
        self._orderParameter = Decimal()

    @classmethod
    def createInstance(cls, parameters: UnimodalProbeInitializerParameters,
                       settings: ProbeSettings, sizer: ProbeSizer,
                       detector: Detector) -> SuperGaussianProbeInitializer:
        initializer = cls(parameters, sizer, detector)
        initializer.syncFromSettings(settings)
        return initializer

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self._annularRadiusInMeters = settings.sgAnnularRadiusInMeters.value
        self._probeWidthInMeters = settings.sgProbeWidthInMeters.value
        self._orderParameter = settings.sgOrderParameter.value
        super().syncFromSettings(settings)

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.sgAnnularRadiusInMeters.value = self._annularRadiusInMeters
        settings.sgProbeWidthInMeters.value = self._probeWidthInMeters
        settings.sgOrderParameter.value = self._orderParameter
        super().syncToSettings(settings)

    @property
    def displayName(self) -> str:
        return 'Super Gaussian'

    @property
    def simpleName(self) -> str:
        return super().simpleName

    def _createPrimaryMode(self) -> ProbeArrayType:
        probeSize_px = self._sizer.getProbeSize()
        gridCenter_px = probeSize_px / 2
        cellCenters_px = numpy.arange(probeSize_px) + 0.5 - gridCenter_px

        # FIXME this should use pixel size at object plane
        Y_px, X_px = numpy.meshgrid(cellCenters_px, cellCenters_px)
        X_m = X_px * float(self._detector.getPixelSizeXInMeters())
        Y_m = Y_px * float(self._detector.getPixelSizeYInMeters())
        R_m = numpy.hypot(X_m, Y_m)

        Z = (R_m - float(self._annularRadiusInMeters)) / float(self._probeWidthInMeters)
        ZP = numpy.power(Z, 2 * float(self._orderParameter))

        return numpy.exp(-ZP / 2) + 0j

    def setAnnularRadiusInMeters(self, value: Decimal) -> None:
        if self._annularRadiusInMeters != value:
            self._annularRadiusInMeters = value
            self.notifyObservers()

    def getAnnularRadiusInMeters(self) -> Decimal:
        return self._annularRadiusInMeters

    def setProbeWidthInMeters(self, value: Decimal) -> None:
        if self._probeWidthInMeters != value:
            self._probeWidthInMeters = value
            self.notifyObservers()

    def getProbeWidthInMeters(self) -> Decimal:
        return self._probeWidthInMeters

    def setOrderParameter(self, value: Decimal) -> None:
        if self._orderParameter != value:
            self._orderParameter = value
            self.notifyObservers()

    def getOrderParameter(self) -> Decimal:
        return max(self._orderParameter, Decimal(1))
