from __future__ import annotations
from decimal import Decimal

from ...api.apparatus import PixelGeometry
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

    def getProbeWavelengthInMeters(self) -> float:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planckConstant_eV_per_Hz = 4.135667696e-15
        lightSpeedInMetersPerSecond = 299792458
        hc_eVm = planckConstant_eV_per_Hz * lightSpeedInMetersPerSecond
        return hc_eVm / float(self._probeSettings.probeEnergyInElectronVolts.value)

    def getLambdaZInSquareMeters(self) -> float:
        lambdaInMeters = self.getProbeWavelengthInMeters()
        zInMeters = self._detector.getDetectorDistanceInMeters()
        return lambdaInMeters * zInMeters

    def getObjectPlanePixelGeometry(self) -> PixelGeometry:
        lambdaZInSquareMeters = self.getLambdaZInSquareMeters()
        extentXInMeters = self._diffractionPatternSizer.getExtentXInPixels() \
                * self._detector.getPixelGeometry().widthInMeters
        extentYInMeters = self._diffractionPatternSizer.getExtentYInPixels() \
                * self._detector.getPixelGeometry().heightInMeters
        return PixelGeometry(
            widthInMeters=lambdaZInSquareMeters / extentXInMeters,
            heightInMeters=lambdaZInSquareMeters / extentYInMeters,
        )

    def getFresnelNumber(self) -> float:
        extentXInMeters = self._diffractionPatternSizer.getExtentXInPixels() \
                * self._detector.getPixelGeometry().widthInMeters
        return extentXInMeters**2 / self.getLambdaZInSquareMeters()

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
        elif observable is self._diffractionPatternSizer:
            self.notifyObservers()


class ApparatusPresenter(Observable, Observer):

    def __init__(self, settings: ProbeSettings, apparatus: Apparatus) -> None:
        super().__init__()
        self._settings = settings
        self._apparatus = apparatus

    @classmethod
    def createInstance(cls, settings: ProbeSettings, apparatus: Apparatus) -> ApparatusPresenter:
        presenter = cls(settings, apparatus)
        settings.addObserver(presenter)
        apparatus.addObserver(presenter)
        return presenter

    def setProbeEnergyInElectronVolts(self, value: Decimal) -> None:
        self._settings.probeEnergyInElectronVolts.value = value

    def getProbeEnergyInElectronVolts(self) -> Decimal:
        return self._settings.probeEnergyInElectronVolts.value

    def getProbeWavelengthInMeters(self) -> Decimal:
        return Decimal.from_float(self._apparatus.getProbeWavelengthInMeters())

    def getObjectPlanePixelGeometry(self) -> PixelGeometry:
        return self._apparatus.getObjectPlanePixelGeometry()

    def getFresnelNumber(self) -> Decimal:
        return Decimal.from_float(self._apparatus.getFresnelNumber())

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._apparatus:
            self.notifyObservers()
