from __future__ import annotations
from decimal import Decimal

from ...api.observer import Observable, Observer
from ..data import CropSizer, Detector
from .settings import ProbeSettings


class Apparatus(Observable, Observer):

    def __init__(self, detector: Detector, cropSizer: CropSizer,
                 probeSettings: ProbeSettings) -> None:
        super().__init__()
        self._detector = detector
        self._cropSizer = cropSizer
        self._probeSettings = probeSettings

    @classmethod
    def createInstance(cls, detector: Detector, cropSizer: CropSizer,
                       probeSettings: ProbeSettings) -> Apparatus:
        sizer = cls(detector, cropSizer, probeSettings)
        detector.addObserver(sizer)
        cropSizer.addObserver(sizer)
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
        extentXInMeters = self._cropSizer.getExtentXInPixels() \
                * self._detector.getPixelSizeXInMeters()
        return self.getLambdaZInSquareMeters() / extentXInMeters

    def getObjectPlanePixelSizeYInMeters(self) -> Decimal:
        extentYInMeters = self._cropSizer.getExtentYInPixels() \
                * self._detector.getPixelSizeYInMeters()
        return self.getLambdaZInSquareMeters() / extentYInMeters

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
        elif observable is self._cropSizer:
            self.notifyObservers()
