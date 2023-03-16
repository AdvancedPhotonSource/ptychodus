from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from ...api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter
from ...api.settings import SettingsRegistry
from ..data import DiffractionPatternSizer
from ..detector import Detector
from ..statefulCore import StateDataType, StatefulCore
from .apparatus import Apparatus, ApparatusPresenter
from .file import FileProbeInitializer
from .fzp import FresnelZonePlateProbeInitializer
from .initializer import ProbeInitializer, UnimodalProbeInitializerParameters
from .probe import Probe
from .settings import ProbeSettings
from .sizer import ProbeSizer
from .superGaussian import SuperGaussianProbeInitializer
from .testPattern import TestPatternProbeInitializer

logger = logging.getLogger(__name__)


class ProbePresenter(Observable, Observer):

    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                 apparatus: Apparatus, initializerChooser: PluginChooser[ProbeInitializer],
                 fileWriterChooser: PluginChooser[ProbeFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._probe = probe
        self._apparatus = apparatus
        self._initializerChooser = initializerChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: ProbeSettings, sizer: ProbeSizer, probe: Probe,
                       apparatus: Apparatus, initializerChooser: PluginChooser[ProbeInitializer],
                       fileWriterChooser: PluginChooser[ProbeFileWriter],
                       reinitObservable: Observable) -> ProbePresenter:
        presenter = cls(settings, sizer, probe, apparatus, initializerChooser, fileWriterChooser,
                        reinitObservable)

        settings.addObserver(presenter)
        sizer.addObserver(presenter)
        probe.addObserver(presenter)
        apparatus.addObserver(presenter)
        reinitObservable.addObserver(presenter)

        presenter._syncFromSettings()

        return presenter

    def isActiveProbeValid(self) -> bool:
        return (self._probe.getProbeExtent() == self._sizer.getProbeExtent())

    def initializeProbe(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Probe')
        initializer.syncToSettings(self._settings)
        self._probe.setArray(initializer())

    def getInitializerNameList(self) -> list[str]:
        return self._initializerChooser.getDisplayNameList()

    def getInitializerName(self) -> str:
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializerByName(self, name: str) -> None:
        self._initializerChooser.setFromDisplayName(name)

    def getInitializer(self) -> ProbeInitializer:
        return self._initializerChooser.getCurrentStrategy()

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
        return self._apparatus.getProbeWavelengthInMeters()

    def setActiveProbe(self, name: str) -> None:
        pass  # TODO

    def getNumberOfProbeModes(self) -> int:
        return self._probe.getNumberOfProbeModes()

    def getProbeModeRelativePower(self, index: int) -> Decimal:
        return self._probe.getProbeModeRelativePower(index)

    def getProbeMode(self, index: int) -> ProbeArrayType:
        return self._probe.getProbeMode(index)

    def _syncFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value)
        initializer = self._initializerChooser.getCurrentStrategy()
        initializer.syncFromSettings(self._settings)
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncFromSettings()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()
        elif observable is self._apparatus:
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self.initializeProbe()


class ProbeCore(StatefulCore):

    @staticmethod
    def _createInitializerChooser(
            rng: numpy.random.Generator, settings: ProbeSettings, sizer: ProbeSizer,
            apparatus: Apparatus,
            fileReaderChooser: PluginChooser[ProbeFileReader]) -> PluginChooser[ProbeInitializer]:
        sgParams = UnimodalProbeInitializerParameters(rng)
        fzpParams = UnimodalProbeInitializerParameters(rng)

        initializerList = [
            FileProbeInitializer.createInstance(settings, sizer, fileReaderChooser),
            SuperGaussianProbeInitializer.createInstance(sgParams, settings, sizer, apparatus),
            FresnelZonePlateProbeInitializer.createInstance(fzpParams, settings, sizer, apparatus),
            TestPatternProbeInitializer.createInstance(sizer),
        ]

        pluginList = [
            PluginEntry[ProbeInitializer](simpleName=ini.simpleName,
                                          displayName=ini.displayName,
                                          strategy=ini) for ini in initializerList
        ]

        return PluginChooser[ProbeInitializer].createFromList(pluginList)

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 detector: Detector, diffractionPatternSizer: DiffractionPatternSizer,
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        self.settings = ProbeSettings.createInstance(settingsRegistry)
        self.sizer = ProbeSizer.createInstance(self.settings, diffractionPatternSizer)
        self.apparatus = Apparatus.createInstance(detector, diffractionPatternSizer, self.settings)
        self.probe = Probe(self.sizer)

        self._initializerChooser = ProbeCore._createInitializerChooser(
            rng, self.settings, self.sizer, self.apparatus, fileReaderChooser)

        self.apparatusPresenter = ApparatusPresenter.createInstance(self.apparatus)
        self.presenter = ProbePresenter.createInstance(self.settings, self.sizer, self.probe,
                                                       self.apparatus, self._initializerChooser,
                                                       fileWriterChooser, settingsRegistry)

    def initializeAndActivateProbe(self) -> None:
        self.presenter.initializeProbe()

    def getStateData(self, *, restartable: bool) -> StateDataType:
        pixelSizeXInMeters = float(self.apparatus.getObjectPlanePixelSizeXInMeters())
        pixelSizeYInMeters = float(self.apparatus.getObjectPlanePixelSizeYInMeters())

        state: StateDataType = {
            'probe': self.probe.getArray(),
            'pixelSizeXInMeters': numpy.array([pixelSizeXInMeters]),
            'pixelSizeYInMeters': numpy.array([pixelSizeYInMeters]),
        }
        return state

    def setStateData(self, state: StateDataType) -> None:
        try:
            array = state['probe']
        except KeyError:
            logger.debug('Failed to restore probe array state!')
        else:
            self.probe.setArray(array)
