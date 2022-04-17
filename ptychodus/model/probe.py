from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Callable
import logging

import numpy
import numpy.typing

from .chooser import StrategyChooser, StrategyEntry
from .crop import CropSizer
from .detector import DetectorSettings
from .fzp import single_probe
from .geometry import Interval
from .image import ImageExtent
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class ProbeSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'GaussianBeam')
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

    def getProbeSize(self) -> int:
        limits = self.getProbeSizeLimits()
        return limits.clamp(self._settings.probeSize.value)

    def getProbeExtent(self) -> ImageExtent:
        size = self.getProbeSize()
        return ImageExtent(width=size, height=size)

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


ComplexNumpyArrayType = numpy.typing.NDArray[numpy.complexfloating]
ProbeInitializerType = Callable[[], ComplexNumpyArrayType]


class GaussianBeamProbeInitializer:
    def __init__(self, detectorSettings: DetectorSettings, probeSettings: ProbeSettings) -> None:
        self._detectorSettings = detectorSettings
        self._probeSettings = probeSettings

    def _createCircularMask(self) -> ComplexNumpyArrayType:
        width_px = self._probeSettings.probeSize.value
        height_px = width_px

        Y_px, X_px = numpy.ogrid[:height_px, :width_px]
        X_m = (X_px - width_px / 2) * float(self._detectorSettings.pixelSizeXInMeters.value)
        Y_m = (Y_px - height_px / 2) * float(self._detectorSettings.pixelSizeYInMeters.value)

        probeRadius_m = self._probeSettings.probeDiameterInMeters.value / 2
        R_m = numpy.hypot(X_m, Y_m)

        return (R_m <= probeRadius_m)

    def __call__(self) -> ComplexNumpyArrayType:
        mask = self._createCircularMask()
        ft = numpy.fft.fft2(mask)
        ft = numpy.fft.fftshift(ft)
        return ft


class FresnelZonePlateProbeInitializer:
    def __init__(self, detectorSettings: DetectorSettings, probeSettings: ProbeSettings,
                 sizer: ProbeSizer) -> None:
        self._detectorSettings = detectorSettings
        self._probeSettings = probeSettings
        self._sizer = sizer

    def __call__(self) -> ComplexNumpyArrayType:
        shape = self._sizer.getProbeSize()
        lambda0 = self._sizer.getWavelengthInMeters()
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


class CustomProbeInitializer(Observer):
    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._array = numpy.zeros(sizer.getProbeExtent().shape, dtype=complex)

    @classmethod
    def createInstance(cls, settings: ProbeSettings, sizer: ProbeSizer) -> CustomProbeInitializer:
        initializer = cls(settings, sizer)
        initializer._openProbeFromSettings()
        settings.customFilePath.addObserver(initializer)
        return initializer

    def __call__(self) -> ComplexNumpyArrayType:
        return self._array

    def getOpenFileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def openProbe(self, filePath: Path) -> None:
        self._settings.customFilePath.value = filePath

    def _openProbeFromSettings(self) -> None:
        customFilePath = self._settings.customFilePath.value

        if customFilePath is not None and customFilePath.is_file():
            logger.debug(f'Reading {customFilePath}')
            self._array = numpy.load(customFilePath)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.customFilePath:
            self._openProbeFromSettings()


class Probe(Observable):
    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._array = numpy.zeros(sizer.getProbeExtent().shape, dtype=complex)

    def getArray(self) -> ComplexNumpyArrayType:
        return self._array

    def setArray(self, array: ComplexNumpyArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Probe must be a complex-valued ndarray')

        self._array = array
        self.notifyObservers()


class ProbeInitializer(Observable, Observer):
    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._probe = probe
        self._reinitObservable = reinitObservable
        self._customInitializer = CustomProbeInitializer.createInstance(settings, sizer)
        self._initializerChooser = StrategyChooser[ProbeInitializerType](
            StrategyEntry[ProbeInitializerType](simpleName='Custom',
                                                displayName='Custom',
                                                strategy=self._customInitializer))

    @classmethod
    def createInstance(cls, detectorSettings: DetectorSettings, probeSettings: ProbeSettings,
                       sizer: ProbeSizer, probe: Probe,
                       reinitObservable: Observable) -> ProbeInitializer:
        initializer = cls(probeSettings, sizer, probe, reinitObservable)

        fzpInit = StrategyEntry[ProbeInitializerType](simpleName='FresnelZonePlate',
                                                      displayName='Fresnel Zone Plate',
                                                      strategy=FresnelZonePlateProbeInitializer(
                                                          detectorSettings, probeSettings, sizer))
        initializer._initializerChooser.addStrategy(fzpInit)

        gaussInit = StrategyEntry[ProbeInitializerType](simpleName='GaussianBeam',
                                                        displayName='Gaussian Beam',
                                                        strategy=GaussianBeamProbeInitializer(
                                                            detectorSettings, probeSettings))
        initializer._initializerChooser.addStrategy(gaussInit)

        probeSettings.initializer.addObserver(initializer)
        initializer._initializerChooser.addObserver(initializer)
        initializer._syncInitializerFromSettings()
        reinitObservable.addObserver(initializer)

        return initializer

    def getInitializerList(self) -> list[str]:
        return self._initializerChooser.getDisplayNameList()

    def getInitializer(self) -> str:
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializer(self, name: str) -> None:
        self._initializerChooser.setFromDisplayName(name)

    def initializeProbe(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        self._probe.setArray(initializer())

    def getOpenFileFilterList(self) -> list[str]:
        return [self._customInitializer.getOpenFileFilter()]

    def openProbe(self, filePath: Path) -> None:
        self._customInitializer.openProbe(filePath)
        self._initializerChooser.setToDefault()
        self.initializeProbe()

    def getSaveFileFilterList(self) -> list[str]:
        return ['NumPy Binary Files (*.npy)']

    def saveProbe(self, filePath: Path) -> None:
        logger.debug(f'Writing {filePath}')
        numpy.save(filePath, self._probe.getArray())

    def _syncInitializerFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value)

    def _syncInitializerToSettings(self) -> None:
        self._settings.initializer.value = self._initializerChooser.getCurrentSimpleName()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self._syncInitializerFromSettings()
        elif observable is self._initializerChooser:
            self._syncInitializerToSettings()
        elif observable is self._reinitObservable:
            self.initializeProbe()


class ProbePresenter(Observable, Observer):
    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                 initializer: ProbeInitializer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._probe = probe
        self._initializer = initializer

    @classmethod
    def createInstance(cls, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                       initializer: ProbeInitializer) -> ProbePresenter:
        presenter = cls(settings, sizer, probe, initializer)
        settings.addObserver(presenter)
        sizer.addObserver(presenter)
        probe.addObserver(presenter)
        initializer.addObserver(presenter)
        return presenter

    def getInitializerList(self) -> list[str]:
        return self._initializer.getInitializerList()

    def getInitializer(self) -> str:
        return self._initializer.getInitializer()

    def setInitializer(self, name: str) -> None:
        self._initializer.setInitializer(name)

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializer.getOpenFileFilterList()

    def openProbe(self, filePath: Path) -> None:
        self._initializer.openProbe(filePath)

    def getSaveFileFilterList(self) -> list[str]:
        return self._initializer.getSaveFileFilterList()

    def saveProbe(self, filePath: Path) -> None:
        self._initializer.saveProbe(filePath)

    def initializeProbe(self) -> None:
        self._initializer.initializeProbe()

    def isAutomaticProbeSizeEnabled(self) -> bool:
        return self._settings.automaticProbeSizeEnabled.value

    def setAutomaticProbeSizeEnabled(self, enabled: bool) -> None:
        self._settings.automaticProbeSizeEnabled.value = enabled

    def getProbeMinSize(self) -> int:
        return self._sizer.getProbeSizeLimits().lower

    def getProbeMaxSize(self) -> int:
        return self._sizer.getProbeSizeLimits().upper

    def setProbeSize(self, value: int) -> None:
        self._settings.probeSize.value = value

    def getProbeSize(self) -> int:
        return self._sizer.getProbeSize()

    def setProbeEnergyInElectronVolts(self, value: Decimal) -> None:
        self._settings.probeEnergyInElectronVolts.value = value

    def getProbeEnergyInElectronVolts(self) -> Decimal:
        return self._settings.probeEnergyInElectronVolts.value

    def getProbeWavelengthInMeters(self) -> Decimal:
        return self._sizer.getWavelengthInMeters()

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

    def getProbe(self) -> ComplexNumpyArrayType:
        return self._probe.getArray()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()
        elif observable is self._initializer:
            self.notifyObservers()
