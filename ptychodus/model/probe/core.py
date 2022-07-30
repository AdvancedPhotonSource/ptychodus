from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from ...api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter, ProbeInitializerType
from ...api.settings import SettingsRegistry, SettingsGroup
from ..data import CropSizer, Detector
from .file import FileProbeInitializer
from .fzp import FresnelZonePlateProbeInitializer
from .probe import Probe
from .settings import ProbeSettings
from .sg import SuperGaussianProbeInitializer
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


class ProbePresenter(Observable, Observer):

    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                 initializerChooser: PluginChooser[ProbeInitializerType],
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._probe = probe
        self._initializerChooser = initializerChooser
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                       initializerChooser: PluginChooser[ProbeInitializerType],
                       fileReaderChooser: PluginChooser[ProbeFileReader],
                       fileWriterChooser: PluginChooser[ProbeFileWriter],
                       reinitObservable: Observable) -> ProbePresenter:
        presenter = cls(settings, sizer, probe, initializerChooser, fileReaderChooser,
                        fileWriterChooser, reinitObservable)

        settings.addObserver(presenter)
        sizer.addObserver(presenter)
        probe.addObserver(presenter)
        initializerChooser.addObserver(presenter)
        fileReaderChooser.addObserver(presenter)
        reinitObservable.addObserver(presenter)

        presenter._syncFromSettings()

        return presenter

    def initializeProbe(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Probe')
        self._probe.setArray(initializer())

    def getInitializerNameList(self) -> list[str]:
        return self._initializerChooser.getDisplayNameList()

    def getInitializer(self) -> str:
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializer(self, name: str) -> None:
        self._initializerChooser.setFromDisplayName(name)

    def getOpenFilePath(self) -> Path:
        return self._settings.inputFilePath.value

    def setOpenFilePath(self, filePath: Path) -> None:
        self._settings.inputFilePath.value = filePath

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def setOpenFileFilter(self, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveProbe(self, filePath: Path, fileFilter: str) -> None:
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        writer = self._fileWriterChooser.getCurrentStrategy()

        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer.write(filePath, self._probe.getArray())

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

    def setSuperGaussianAnnularRadiusInMeters(self, value: Decimal) -> None:
        self._settings.sgAnnularRadiusInMeters.value = value

    def getSuperGaussianAnnularRadiusInMeters(self) -> Decimal:
        return self._settings.sgAnnularRadiusInMeters.value

    def setSuperGaussianProbeWidthInMeters(self, value: Decimal) -> None:
        self._settings.sgProbeWidthInMeters.value = value

    def getSuperGaussianProbeWidthInMeters(self) -> Decimal:
        return self._settings.sgProbeWidthInMeters.value

    def setSuperGaussianOrderParameter(self, value: Decimal) -> None:
        self._settings.sgOrderParameter.value = value

    def getSuperGaussianOrderParameter(self) -> Decimal:
        return max(self._settings.sgOrderParameter.value, Decimal(1))

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

    def getProbeModeRelativePower(self, index: int) -> Decimal:
        return self._probe.getProbeModeRelativePower(index)

    def getProbeMode(self, index: int) -> ProbeArrayType:
        return self._probe.getProbeMode(index)

    def pushProbeMode(self) -> None:
        self._probe.pushProbeMode()

    def popProbeMode(self) -> None:
        self._probe.popProbeMode()

    def _syncFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value)
        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)
        self.notifyObservers()

    def _syncInitializerToSettings(self) -> None:
        self._settings.initializer.value = self._initializerChooser.getCurrentSimpleName()

    def _syncFileReaderToSettings(self) -> None:
        self._settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncFromSettings()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()
        elif observable is self._initializerChooser:
            self._syncInitializerToSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._reinitObservable:
            self.initializeProbe()


class ProbeCore:

    def __init__(self, settingsRegistry: SettingsRegistry, detector: Detector,
                 cropSizer: CropSizer, fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        self.settings = ProbeSettings.createInstance(settingsRegistry)
        self.sizer = ProbeSizer.createInstance(self.settings, cropSizer)
        self.probe = Probe(self.settings, self.sizer)

        self._filePlugin = PluginEntry[ProbeInitializerType](
            simpleName='FromFile',
            displayName='Open File...',
            strategy=FileProbeInitializer(self.settings, self.sizer, fileReaderChooser),
        )
        self._sgPlugin = PluginEntry[ProbeInitializerType](
            simpleName='SuperGaussian',
            displayName='Super Gaussian',
            strategy=SuperGaussianProbeInitializer(detector, self.settings),
        )
        self._fzpPlugin = PluginEntry[ProbeInitializerType](
            simpleName='FresnelZonePlate',
            displayName='Fresnel Zone Plate',
            strategy=FresnelZonePlateProbeInitializer(detector, self.settings, self.sizer),
        )
        self._initializerChooser = PluginChooser[ProbeInitializerType].createFromList(
            [self._filePlugin, self._sgPlugin, self._fzpPlugin])

        self.presenter = ProbePresenter.createInstance(self.settings, self.sizer, self.probe,
                                                       self._initializerChooser, fileReaderChooser,
                                                       fileWriterChooser, settingsRegistry)
