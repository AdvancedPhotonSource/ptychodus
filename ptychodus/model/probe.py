from __future__ import annotations
from decimal import Decimal
from typing import Callable

import numpy

from .detector import DetectorSettings
from .fzp import single_probe
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup


class ProbeSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Gaussian')
        self.customFilePath = settingsGroup.createPathEntry('CustomFilePath', None)
        self.probeShape = settingsGroup.createIntegerEntry('ProbeShape', 64)
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
    def createInstance(cls, settingsRegistry: SettingsRegistry):
        settings = cls(settingsRegistry.createGroup('Probe'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class Probe(Observable, Observer):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: ProbeSettings) -> Probe:
        probe = cls(settings)
        settings.probeShape.addObserver(probe)
        settings.probeEnergyInElectronVolts.addObserver(probe)
        return probe

    @property
    def extentInPixels(self) -> int:
        return self._settings.probeShape.value

    @property
    def wavelengthInMeters(self) -> Decimal:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planck_constant_eV_per_Hz = Decimal(4.135667696e-15)
        light_speed_m_per_s = Decimal(299792458)
        hc_eVm = planck_constant_eV_per_Hz * light_speed_m_per_s
        probe_wavelength_m = hc_eVm / self._settings.probeEnergyInElectronVolts.value
        return probe_wavelength_m

    def update(self, observable: Observable) -> None:
        if observable is self._settings.probeShape:
            self.notifyObservers()
        elif observable is self._settings.probeEnergyInElectronVolts:
            self.notifyObservers()


class ProbeIO:
    FILE_FILTER = 'NumPy Binary Files (*.npy)'

    def write(self, filePath: Path, estimate: numpy.ndarray) -> None:
        numpy.save(filePath, estimate)

    def read(self, filePath: Path) -> numpy.ndarray:
        return numpy.load(filePath)


class GaussianBeamProbeInitializer(Callable):
    def __init__(self, detectorSettings: DetectorSettings, probeSettings: ProbeSettings) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._probeSettings = probeSettings

    def __call__(self) -> numpy.ndarray:
        probeShape = self._probeSettings.probeShape.value
        idx = numpy.arange(probeShape) + (1 - probeShape) / 2
        w_m = self._probeSettings.probeDiameterInMeters.value
        x = idx * float(self._detectorSettings.pixelSizeXInMeters.value / w_m)
        y = idx * float(self._detectorSettings.pixelSizeYInMeters.value / w_m)
        xx, yy = numpy.meshgrid(x, y)
        rr_sq = xx**2 + yy**2
        ff = numpy.exp(-rr_sq / 2) + 0.j
        return ff / ff.sum()

    def __str__(self) -> str:
        return 'Gaussian Beam'


class FresnelZonePlateProbeInitializer(Callable):
    def __init__(self, detectorSettings: DetectorSettings, probeSettings: ProbeSettings,
                 probe: Probe) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._probeSettings = probeSettings
        self._probe = probe

    def __call__(self) -> numpy.ndarray:
        shape = self._probeSettings.probeShape.value
        lambda0 = self._probe.wavelengthInMeters
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


class CustomProbeInitializer(Callable):
    def __init__(self) -> None:
        super().__init__()
        self._initialProbe = numpy.zeros((64, 64), dtype=complex)

    def setInitialProbe(self, initialProbe: numpy.ndarray) -> None:
        if not numpy.iscomplexobj(initialProbe):
            raise TypeError('Probe must be a complex-valued ndarray')

        self._initialProbe = initialProbe

    def __call__(self) -> numpy.ndarray:
        return self._initialProbe

    def __str__(self) -> str:
        return 'Custom'


class ProbePresenter(Observable, Observer):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: ProbeSettings, probe: Probe,
                 initializerList: list[Callable]) -> None:
        super().__init__()
        self._settings = settings
        self._probe = probe
        self._initializerList = initializerList
        self._initializer = initializerList[0]
        self._estimate = numpy.zeros((0, 0), dtype=complex)
        self._probeIO = ProbeIO()

    @classmethod
    def createInstance(cls, detectorSettings: DetectorSettings, probeSettings: ProbeSettings,
                       probe: Probe) -> ProbePresenter:
        initializerList = list()
        initializerList.append(GaussianBeamProbeInitializer(detectorSettings, probeSettings))
        initializerList.append(
            FresnelZonePlateProbeInitializer(detectorSettings, probeSettings, probe))
        initializerList.append(CustomProbeInitializer())

        presenter = cls(probeSettings, probe, initializerList)
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
        self._settings.customFilePath.value = filePath
        initialProbe = self._probeIO.read(filePath)
        self.setCurrentInitializer('Custom')
        self._initializer.setInitialProbe(initialProbe)
        self.initializeProbe()

    def saveProbe(self, filePath: Path) -> None:
        self._probeIO.write(filePath, self._estimate)

    def initializeProbe(self) -> None:
        self._estimate = self._initializer()
        self.notifyObservers()

    def getProbeMinShape(self) -> int:
        return 0

    def getProbeMaxShape(self) -> int:
        return self.MAX_INT

    def setProbeShape(self, value: int) -> None:
        self._settings.probeShape.value = value

    def getProbeShape(self) -> int:
        valueMin = self.getProbeMinShape()
        valueMax = self.getProbeMaxShape()
        value = self._settings.probeShape.value
        valueClamp = max(valueMin, min(value, valueMax))
        return valueClamp

    def setProbeEnergyInElectronVolts(self, value: Decimal) -> None:
        self._settings.probeEnergyInElectronVolts.value = value

    def getProbeEnergyInElectronVolts(self) -> Decimal:
        return self._settings.probeEnergyInElectronVolts.value

    def getProbeWavelengthInMeters(self) -> Decimal:
        return self._probe.wavelengthInMeters

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
        return self._estimate

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setCurrentInitializerFromSettings()
        elif observable is self._settings:
            self.notifyObservers()
