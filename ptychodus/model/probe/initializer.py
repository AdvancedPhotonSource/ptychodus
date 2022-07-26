from __future__ import annotations
from pathlib import Path
import logging

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from ...api.probe import ProbeFileWriter, ProbeInitializerType
from ..data import Detector
from .file import FileProbeInitializer
from .fzp import FresnelZonePlateProbeInitializer
from .probe import Probe
from .settings import ProbeSettings
from .sg import SuperGaussianProbeInitializer
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


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

        # FIXME need to update so that FromFile does not show in GUI
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

        gaussInit = PluginEntry[ProbeInitializerType](simpleName='SuperGaussian',
                                                      displayName='Super Gaussian',
                                                      strategy=SuperGaussianProbeInitializer(
                                                          detector, probeSettings))
        initializer._initializerChooser.addStrategy(gaussInit)

        probeSettings.initializer.addObserver(initializer)
        initializer._initializerChooser.addObserver(initializer)
        initializer._syncInitializerFromSettings()
        reinitObservable.addObserver(initializer)

        return initializer

    def getInitializerNameList(self) -> list[str]:
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

    def getOpenFileFilter(self) -> str:
        return self._fileInitializer.getOpenFileFilter()

    def openProbe(self, filePath: Path, fileFilter: str) -> None:
        self._fileInitializer.openProbe(filePath, fileFilter)
        self._initializerChooser.setToDefault()
        self.initializeProbe()

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveProbe(self, filePath: Path, fileFilter: str) -> None:
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
