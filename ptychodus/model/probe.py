from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Callable

import numpy

from .crop import CropSizer
from .detector import DetectorSettings
from .fzp import single_probe
from .geometry import Interval
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup


class ProbeSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Gaussian')
        self.customFilePath = settingsGroup.createPathEntry('CustomFilePath', None)
        self.automaticProbeSizeEnabled = settingsGroup.createBooleanEntry(
            'AutomaticProbeSizeEnabled', True)
        self.probeSize = settingsGroup.createIntegerEntry('ProbeSize', 64)
        self.probeEnergyInElectronVolts = settingsGroup.createRealEntry(
            'ProbeEnergyInElectronVolts', '2000')
        self.probeDiameterInMeters = settingsGroup.createRealEntry('ProbeDiameterInMeters',
                                                                   '400e-6')
        self.zonePlateRadiusInMeters = settingsGroup.createRealEntry('ZonePlateRadiusInMeters',
                                                                     '90e-6')
        self.outermostZoneWidthInMeters = settingsGroup.createRealEntry(
            'OutermostZoneWidthInMeters', '50e-9')
        self.beamstopDiameterInMeters = settingsGroup.createRealEntry(
            'BeamstopDiameterInMeters', '60e-6')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ProbeSettings:
        settings = cls(settingsRegistry.createGroup('Probe'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class ProbeSizer(Observable, Observer):
    def __init__(self, settings: ProbeSettings, cropSizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._cropSizer = cropSizer

    @classmethod
    def createInstance(cls, settings: ProbeSettings, cropSizer: CropSizer) -> ProbeSizer:
        sizer = cls(settings, cropSizer)
        settings.addObserver(sizer)
        cropSizer.addObserver(sizer)
        return sizer

    @property
    def _probeSizeMax(self) -> int:
        cropX = self._cropSizer.getExtentX()
        cropY = self._cropSizer.getExtentY()
        return min(cropX, cropY)

    def getProbeSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self._probeSizeMax)

    def getProbeSize(self) -> int:  # FIXME shouldn't this clamp?
        return self._settings.probeSize.value

    def getWavelengthInMeters(self) -> Decimal:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planck_constant_eV_per_Hz = Decimal(4.135667696e-15)
        light_speed_m_per_s = Decimal(299792458)
        hc_eVm = planck_constant_eV_per_Hz * light_speed_m_per_s
        probe_wavelength_m = hc_eVm / self._settings.probeEnergyInElectronVolts.value
        return probe_wavelength_m

    def _updateProbeSize(self) -> None:
        if self._settings.automaticProbeSizeEnabled.value:
            self._settings.probeSize.value = self._probeSizeMax

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._updateProbeSize()
        elif observable is self._cropSizer:
            self._updateProbeSize()


class Probe(Observable):
    FILE_FILTER = 'NumPy Binary Files (*.npy)'

    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__()
        self._settings = settings
        self._array = numpy.zeros((0, 0), dtype=complex)

    @classmethod
    def createInstance(cls, settings: ProbeSettings) -> Probe:
        return cls(settings)

    def getArray(self) -> numpy.ndarray:
        return self._array

    def setArray(self, array: numpy.ndarray) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Probe must be a complex-valued ndarray')

        self._array = array
        self.notifyObservers()

    def read(self, filePath: Path) -> None:
        self._settings.customFilePath.value = filePath
        array = numpy.load(filePath)
        self.setArray(array)

    def write(self, filePath: Path) -> None:
        numpy.save(filePath, self._array)


class GaussianBeamProbeInitializer:
    def __init__(self, detectorSettings: DetectorSettings, probeSettings: ProbeSettings) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._probeSettings = probeSettings

    def _createCircularMask(self) -> numpy.ndarray:
        width_px = self._probeSettings.probeSize.value
        height_px = width_px

        Y_px, X_px = numpy.ogrid[:height_px, :width_px]
        X_m = (X_px - width_px / 2) * float(self._detectorSettings.pixelSizeXInMeters.value)
        Y_m = (Y_px - height_px / 2) * float(self._detectorSettings.pixelSizeYInMeters.value)

        probeRadius_m = self._probeSettings.probeDiameterInMeters.value / 2
        R_m = numpy.hypot(X_m, Y_m)

        return (R_m <= probeRadius_m)

    def __call__(self) -> numpy.ndarray:
        mask = self._createCircularMask()
        ft = numpy.fft.fft2(mask)
        ft = numpy.fft.fftshift(ft)
        return ft

    def __str__(self) -> str:
        return 'Gaussian Beam'


class FresnelZonePlateProbeInitializer:
    def __init__(self, detectorSettings: DetectorSettings, probeSettings: ProbeSettings,
                 probeSizer: ProbeSizer) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._probeSettings = probeSettings
        self._probeSizer = probeSizer

    def __call__(self) -> numpy.ndarray:
        shape = self._probeSizer.getProbeSize()
        lambda0 = self._probeSizer.getWavelengthInMeters()
        dx_dec = self._detectorSettings.pixelSizeXInMeters.value  # TODO non-square pixels are unsupported
        dis_defocus = self._detectorSettings.defocusDistanceInMeters.value
        dis_StoD = self._detectorSettings.detectorDistanceInMeters.value
        radius = self._probeSettings.zonePlateRadiusInMeters.value
        outmost = self._probeSettings.outermostZoneWidthInMeters.value
        beamstop = self._probeSettings.beamstopDiameterInMeters.value

        probe = single_probe(shape,
                             float(lambda0),
                             float(dx_dec),
                             float(dis_defocus),
                             float(dis_StoD),
                             radius=float(radius),
                             outmost=float(outmost),
                             beamstop=float(beamstop))
        return probe

    def __str__(self) -> str:
        return 'Fresnel Zone Plate'


class CustomProbeInitializer:
    def __init__(self) -> None:
        super().__init__()
        self._initialProbe = numpy.zeros((64, 64), dtype=complex)

    def cacheProbe(self, probe: numpy.ndarray) -> None:
        if not numpy.iscomplexobj(probe):
            raise TypeError('Probe must be a complex-valued ndarray')

        self._probe = probe

    def __call__(self) -> numpy.ndarray:
        return self._initialProbe

    def __str__(self) -> str:
        return 'Custom'


class ProbePresenter(Observable, Observer):
    def __init__(self, settings: ProbeSettings, probeSizer: ProbeSizer, probe: Probe,
                 initializerList: Sequence[Callable]) -> None:
        super().__init__()
        self._settings = settings
        self._probeSizer = probeSizer
        self._probe = probe
        self._initializerList = initializerList
        self._initializer = initializerList[0]

    @classmethod
    def createInstance(cls, detectorSettings: DetectorSettings, probeSettings: ProbeSettings,
                       probeSizer: ProbeSizer, probe: Probe) -> ProbePresenter:
        initializerList = list()
        initializerList.append(GaussianBeamProbeInitializer(detectorSettings, probeSettings))
        initializerList.append(
            FresnelZonePlateProbeInitializer(detectorSettings, probeSettings, probeSizer))
        initializerList.append(CustomProbeInitializer())

        presenter = cls(probeSettings, probeSizer, probe, initializerList)
        presenter.setCurrentInitializerFromSettings()
        probeSettings.initializer.addObserver(presenter)
        probeSettings.addObserver(presenter)

        return presenter

    def getInitializerList(self) -> list[str]:
        return [str(initializer) for initializer in self._initializerList]

    def getCurrentInitializer(self) -> str:
        return str(self._initializer)

    def setCurrentInitializer(self, name: str) -> None:
        try:
            initializer = next(ini for ini in self._initializerList
                               if name.casefold() == str(ini).casefold())
        except StopIteration:
            return

        if initializer is not self._initializer:
            self._initializer = initializer
            self._settings.initializer.value = str(self._initializer)
            self.notifyObservers()

    def setCurrentInitializerFromSettings(self) -> None:
        self.setCurrentInitializer(self._settings.initializer.value)

    def openProbe(self, filePath: Path) -> None:
        self._probe.read(filePath)
        self.setCurrentInitializer('Custom')
        self._initializer.cacheProbe(self._probe.getArray())
        self.notifyObservers()

    def saveProbe(self, filePath: Path) -> None:
        self._probe.write(filePath)

    def initializeProbe(self) -> None:
        self._probe.setArray(self._initializer())
        self.notifyObservers()

    def isAutomaticProbeSizeEnabled(self) -> bool:
        return self._settings.automaticProbeSizeEnabled.value

    def setAutomaticProbeSizeEnabled(self, enabled: bool) -> None:
        self._settings.automaticProbeSizeEnabled.value = enabled

    def getProbeMinSize(self) -> int:
        return self._probeSizer.getProbeSizeLimits().lower

    def getProbeMaxSize(self) -> int:
        return self._probeSizer.getProbeSizeLimits().upper

    def setProbeSize(self, value: int) -> None:
        self._settings.probeSize.value = value

    def getProbeSize(self) -> int:
        return self._probeSizer.getProbeSize()

    def setProbeEnergyInElectronVolts(self, value: Decimal) -> None:
        self._settings.probeEnergyInElectronVolts.value = value

    def getProbeEnergyInElectronVolts(self) -> Decimal:
        return self._settings.probeEnergyInElectronVolts.value

    def getProbeWavelengthInMeters(self) -> Decimal:
        return self._probeSizer.getWavelengthInMeters()

    def setProbeDiameterInMeters(self, value: Decimal) -> None:
        self._settings.probeDiameterInMeters.value = value

    def getProbeDiameterInMeters(self) -> Decimal:
        return self._settings.probeDiameterInMeters.value

    def setZonePlateRadiusInMeters(self, value: Decimal) -> None:
        self._settings.zonePlateRadiusInMeters.value = value

    def getZonePlateRadiusInMeters(self) -> Decimal:
        return self._settings.zonePlateRadiusInMeters.value

    def setOutermostZoneWidthInMeters(self, value: Decimal) -> None:
        self._settings.outermostZoneWidthInMeters.value = value

    def getOutermostZoneWidthInMeters(self) -> Decimal:
        return self._settings.outermostZoneWidthInMeters.value

    def setBeamstopDiameterInMeters(self, value: Decimal) -> None:
        self._settings.beamstopDiameterInMeters.value = value

    def getBeamstopDiameterInMeters(self) -> Decimal:
        return self._settings.beamstopDiameterInMeters.value

    def getProbe(self) -> numpy.ndarray:
        return self._probe.getArray()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setCurrentInitializerFromSettings()
        elif observable is self._settings:
            self.notifyObservers()
