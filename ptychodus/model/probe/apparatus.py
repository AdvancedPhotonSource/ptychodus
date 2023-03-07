from __future__ import annotations
from decimal import Decimal

from ...api.observer import Observable, Observer
from ..data import DiffractionPatternSizer
from ..detector import Detector
from .settings import ProbeSettings


class Apparatus(Observable, Observer):

    def __init__(self, detector: Detector, diffractionPatternSizer: DiffractionPatternSizer,
                 probeSettings: ProbeSettings) -> None:
        super().__init__()
        self._detector = detector
        self._diffractionPatternSizer = diffractionPatternSizer
        self._probeSettings = probeSettings

    @classmethod
    def createInstance(cls, detector: Detector, diffractionPatternSizer: DiffractionPatternSizer,
                       probeSettings: ProbeSettings) -> Apparatus:
        sizer = cls(detector, diffractionPatternSizer, probeSettings)
        detector.addObserver(sizer)
        diffractionPatternSizer.addObserver(sizer)
        probeSettings.addObserver(sizer)
        return sizer

    def getProbeWavelengthInMeters(self) -> Decimal:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planckConstant_eV_per_Hz = Decimal('4.135667696e-15')
        lightSpeedInMetersPerSecond = Decimal(299792458)
        hc_eVm = planckConstant_eV_per_Hz * lightSpeedInMetersPerSecond
        return hc_eVm / self._probeSettings.probeEnergyInElectronVolts.value

    def getLambdaZInSquareMeters(self) -> Decimal:
        lambdaInMeters = self.getProbeWavelengthInMeters()
        zInMeters = self._detector.getDetectorDistanceInMeters()
        return lambdaInMeters * zInMeters

    def getObjectPlanePixelSizeXInMeters(self) -> Decimal:
        extentXInMeters = self._diffractionPatternSizer.getExtentXInPixels() \
                * self._detector.getPixelSizeXInMeters()
        return self.getLambdaZInSquareMeters() / extentXInMeters

    def getObjectPlanePixelSizeYInMeters(self) -> Decimal:
        extentYInMeters = self._diffractionPatternSizer.getExtentYInPixels() \
                * self._detector.getPixelSizeYInMeters()
        return self.getLambdaZInSquareMeters() / extentYInMeters

    def getFresnelNumber(self) -> Decimal:
        extentXInMeters = self._diffractionPatternSizer.getExtentXInPixels() \
                * self._detector.getPixelSizeXInMeters()
        return extentXInMeters**2 / self.getLambdaZInSquareMeters()

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
        elif observable is self._diffractionPatternSizer:
            self.notifyObservers()


class ApparatusPresenter(Observable, Observer):

    def __init__(self, apparatus: Apparatus) -> None:
        super().__init__()
        self._apparatus = apparatus

    @classmethod
    def createInstance(cls, apparatus: Apparatus) -> ApparatusPresenter:
        presenter = cls(apparatus)
        apparatus.addObserver(presenter)
        return presenter

    def getFresnelNumber(self) -> Decimal:
        return self._apparatus.getFresnelNumber()

    def update(self, observable: Observable) -> None:
        if observable is self._apparatus:
            self.notifyObservers()
