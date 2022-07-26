from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.probe import *
from ...api.settings import SettingsRegistry, SettingsGroup
from ..data import CropSizer, Detector
from .file import FileProbeInitializer
from .initializer import ProbeInitializer
from .probe import Probe
from .settings import ProbeSettings
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


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

    def getInitializerNameList(self) -> list[str]:
        return self._initializer.getInitializerNameList()

    def getInitializer(self) -> str:
        return self._initializer.getInitializer()

    def setInitializer(self, name: str) -> None:
        self._initializer.setInitializer(name)

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializer.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._initializer.getOpenFileFilter()

    def openProbe(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.openProbe(filePath, fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
        return self._initializer.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._initializer.getSaveFileFilter()

    def saveProbe(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.saveProbe(filePath, fileFilter)

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

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()
        elif observable is self._initializer:
            self.notifyObservers()


class ProbeCore:

    def __init__(self, settingsRegistry: SettingsRegistry, detector: Detector,
                 cropSizer: CropSizer, fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        self.settings = ProbeSettings.createInstance(settingsRegistry)
        self.sizer = ProbeSizer.createInstance(self.settings, cropSizer)
        self.probe = Probe(self.settings, self.sizer)

        self._fileInitializer = FileProbeInitializer.createInstance(self.settings, self.sizer,
                                                                    fileReaderChooser)
        self._initializer = ProbeInitializer.createInstance(detector, self.settings, self.sizer,
                                                            self.probe, self._fileInitializer,
                                                            fileWriterChooser, settingsRegistry)

        self.presenter = ProbePresenter.createInstance(self.settings, self.sizer, self.probe,
                                                       self._initializer)
