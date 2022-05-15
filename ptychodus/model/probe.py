from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
import logging

import numpy

from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry
from ..api.probe import *
from ..api.settings import SettingsRegistry, SettingsGroup
from .detector import CropSizer, Detector
from .fzp import single_probe
from .geometry import Interval
from .image import ImageExtent

logger = logging.getLogger(__name__)


class ProbeSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'GaussianBeam')
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'CSV')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath', None)
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
        self.defocusDistanceInMeters = settingsGroup.createRealEntry('DefocusDistanceInMeters',
                                                                     '800e-6')

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


class GaussianBeamProbeInitializer:
    def __init__(self, detector: Detector, probeSettings: ProbeSettings) -> None:
        self._detector = detector
        self._probeSettings = probeSettings

    def _createCircularMask(self) -> ProbeArrayType:
        width_px = self._probeSettings.probeSize.value
        height_px = width_px

        Y_px, X_px = numpy.ogrid[:height_px, :width_px]
        X_m = (X_px - width_px / 2) * float(self._detector.pixelSizeXInMeters)
        Y_m = (Y_px - height_px / 2) * float(self._detector.pixelSizeYInMeters)

        probeRadius_m = self._probeSettings.probeDiameterInMeters.value / 2
        R_m = numpy.hypot(X_m, Y_m)

        return (R_m <= probeRadius_m)

    def __call__(self) -> ProbeArrayType:
        mask = self._createCircularMask()
        ft = numpy.fft.fft2(mask)
        ft = numpy.fft.fftshift(ft)
        return ft


class FresnelZonePlateProbeInitializer:
    def __init__(self, detector: Detector, probeSettings: ProbeSettings,
                 sizer: ProbeSizer) -> None:
        self._detector = detector
        self._probeSettings = probeSettings
        self._sizer = sizer

    def __call__(self) -> ProbeArrayType:
        shape = self._sizer.getProbeSize()
        lambda0 = self._sizer.getWavelengthInMeters()
        dx_dec = self._detector.pixelSizeXInMeters  # TODO non-square pixels are unsupported
        dis_defocus = self._probeSettings.defocusDistanceInMeters.value
        dis_StoD = self._detector.distanceToObjectInMeters
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


class FileProbeInitializer(Observer):
    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer,
                 fileReaderChooser: PluginChooser[ProbeFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._array = numpy.zeros(sizer.getProbeExtent().shape, dtype=complex)

    @classmethod
    def createInstance(cls, settings: ProbeSettings, sizer: ProbeSizer,
                       fileReaderChooser: PluginChooser[ProbeFileReader]) -> FileProbeInitializer:
        initializer = cls(settings, sizer, fileReaderChooser)

        settings.inputFileType.addObserver(initializer)
        initializer._fileReaderChooser.addObserver(initializer)
        initializer._syncFileReaderFromSettings()

        settings.inputFilePath.addObserver(initializer)
        initializer._openProbeFromSettings()

        return initializer

    def __call__(self) -> ProbeArrayType:
        return self._array

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openProbe(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            self._array = fileReader.read(filePath)

    def openProbe(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.inputFilePath.value == filePath:
            self._openProbe(filePath)

        self._settings.inputFilePath.value = filePath

    def _openProbeFromSettings(self) -> None:
        self._openProbe(self._settings.inputFilePath.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.inputFileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.inputFilePath:
            self._openProbeFromSettings()


class Probe(Observable):
    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._array = numpy.zeros((1, *sizer.getProbeExtent().shape), dtype=complex)

    def getNumberOfProbeModes(self) -> int:
        return self._array.shape[0]

    def getProbeMode(self, index: int) -> ProbeArrayType:
        return self._array[index, ...]

    def getArray(self) -> ProbeArrayType:
        return self._array

    def setArray(self, array: ProbeArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Probe must be a complex-valued ndarray')

        if array.ndim == 2:
            self._array = array[numpy.newaxis, ...]
        elif array.ndim == 3:
            self._array = array
        else:
            raise ValueError('Probe must be 2- or 3-dimensional ndarray.')

        self.notifyObservers()


class ProbeInitializer(Observable, Observer):
    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                 fileInitializer: FileProbeInitializer,
                 fileWriterChooser: PluginChooser[ProbeFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._probe = probe
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable
        self._fileInitializer = fileInitializer
        self._initializerChooser = PluginChooser[ProbeInitializerType](
            PluginEntry[ProbeInitializerType](simpleName='FromFile',
                                              displayName='From File',
                                              strategy=self._fileInitializer))

    @classmethod
    def createInstance(cls, detector: Detector, probeSettings: ProbeSettings, sizer: ProbeSizer,
                       probe: Probe, fileInitializer: FileProbeInitializer,
                       fileWriterChooser: PluginChooser[ProbeFileWriter],
                       reinitObservable: Observable) -> ProbeInitializer:
        initializer = cls(probeSettings, sizer, probe, fileInitializer, fileWriterChooser,
                          reinitObservable)

        fzpInit = PluginEntry[ProbeInitializerType](simpleName='FresnelZonePlate',
                                                    displayName='Fresnel Zone Plate',
                                                    strategy=FresnelZonePlateProbeInitializer(
                                                        detector, probeSettings, sizer))
        initializer._initializerChooser.addStrategy(fzpInit)

        gaussInit = PluginEntry[ProbeInitializerType](simpleName='GaussianBeam',
                                                      displayName='Gaussian Beam',
                                                      strategy=GaussianBeamProbeInitializer(
                                                          detector, probeSettings))
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
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Probe')
        self._probe.setArray(initializer())

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileInitializer.getOpenFileFilterList()

    def openProbe(self, filePath: Path) -> None:
        self._fileInitializer.openProbe(filePath)
        self._initializerChooser.setToDefault()
        self.initializeProbe()

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def saveProbe(self, filePath: Path) -> None:
        logger.debug(f'Writing {filePath}')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, self._probe.getArray())

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

    def getDefocusDistanceInMeters(self) -> Decimal:
        return self._settings.defocusDistanceInMeters.value

    def setDefocusDistanceInMeters(self, value: Decimal) -> None:
        self._settings.defocusDistanceInMeters.value = value

    def getNumberOfProbeModes(self) -> int:
        return self._probe.getNumberOfProbeModes()

    def getProbeMode(self, index: int) -> ProbeArrayType:
        return self._probe.getProbeMode(index)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()
        elif observable is self._initializer:
            self.notifyObservers()
