from __future__ import annotations
from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Generator, Optional
import logging

import numpy
import numpy.typing

from ...api.geometry import Interval
from ...api.object import ObjectPhaseCenteringStrategy
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
from ...api.settings import SettingsRegistry
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..probe import ProbeAPI
from ..scan import ScanAPI
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings
from .trainable import TrainableReconstructor

logger = logging.getLogger(__name__)


class PtychoNNModelPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoNNModelSettings) -> None:
        super().__init__()
        self._settings = settings
        self._fileFilterList: list[str] = ['PyTorch Model State Files (*.pt *.pth)']

    @classmethod
    def createInstance(cls, settings: PtychoNNModelSettings) -> PtychoNNModelPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getStateFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getStateFileFilter(self) -> str:
        return self._fileFilterList[0]

    def getStateFilePath(self) -> Path:
        return self._settings.stateFilePath.value

    def setStateFilePath(self, directory: Path) -> None:
        self._settings.stateFilePath.value = directory

    def getNumberOfConvolutionChannelsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfConvolutionChannels(self) -> int:
        limits = self.getNumberOfConvolutionChannelsLimits()
        return limits.clamp(self._settings.numberOfConvolutionChannels.value)

    def setNumberOfConvolutionChannels(self, value: int) -> None:
        self._settings.numberOfConvolutionChannels.value = value

    def getBatchSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getBatchSize(self) -> int:
        limits = self.getBatchSizeLimits()
        return limits.clamp(self._settings.batchSize.value)

    def setBatchSize(self, value: int) -> None:
        self._settings.batchSize.value = value

    def isBatchNormalizationEnabled(self) -> bool:
        return self._settings.useBatchNormalization.value

    def setBatchNormalizationEnabled(self, enabled: bool) -> None:
        self._settings.useBatchNormalization.value = enabled

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoNNTrainingPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoNNTrainingSettings,
                 phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                 diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                 probeAPI: ProbeAPI, objectAPI: ObjectAPI, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._phaseCenteringStrategyChooser = phaseCenteringStrategyChooser
        self._diffractionDataset = diffractionDataset
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._reinitObservable = reinitObservable
        self._trainer: Optional[TrainableReconstructor] = None
        self._fileFilterList: list[str] = ['NumPy Zipped Archive (*.npz)']

    @classmethod
    def createInstance(cls, settings: PtychoNNTrainingSettings,
                       phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                       diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                       probeAPI: ProbeAPI, objectAPI: ObjectAPI,
                       reinitObservable: Observable) -> PtychoNNTrainingPresenter:
        presenter = cls(settings, phaseCenteringStrategyChooser, diffractionDataset, scanAPI,
                        probeAPI, objectAPI, reinitObservable)
        reinitObservable.addObserver(presenter)
        phaseCenteringStrategyChooser.addObserver(presenter)
        presenter._syncFromSettings()
        return presenter

    def setTrainer(self, trainer: TrainableReconstructor) -> None:
        self._trainer = trainer

    def getValidationSetFractionalSizeLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getValidationSetFractionalSize(self) -> Decimal:
        limits = self.getValidationSetFractionalSizeLimits()
        return limits.clamp(self._settings.validationSetFractionalSize.value)

    def setValidationSetFractionalSize(self, value: Decimal) -> None:
        self._settings.validationSetFractionalSize.value = value

    def getOptimizationEpochsPerHalfCycleLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getOptimizationEpochsPerHalfCycle(self) -> int:
        limits = self.getOptimizationEpochsPerHalfCycleLimits()
        return limits.clamp(self._settings.optimizationEpochsPerHalfCycle.value)

    def setOptimizationEpochsPerHalfCycle(self, value: int) -> None:
        self._settings.optimizationEpochsPerHalfCycle.value = value

    def getMaximumLearningRateLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getMaximumLearningRate(self) -> Decimal:
        limits = self.getMaximumLearningRateLimits()
        return limits.clamp(self._settings.maximumLearningRate.value)

    def setMaximumLearningRate(self, value: Decimal) -> None:
        self._settings.maximumLearningRate.value = value

    def getMinimumLearningRateLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getMinimumLearningRate(self) -> Decimal:
        limits = self.getMinimumLearningRateLimits()
        return limits.clamp(self._settings.minimumLearningRate.value)

    def setMinimumLearningRate(self, value: Decimal) -> None:
        self._settings.minimumLearningRate.value = value

    def getTrainingEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getTrainingEpochs(self) -> int:
        limits = self.getTrainingEpochsLimits()
        return limits.clamp(self._settings.trainingEpochs.value)

    def setTrainingEpochs(self, value: int) -> None:
        self._settings.trainingEpochs.value = value

    def isSaveTrainingArtifactsEnabled(self) -> bool:
        return self._settings.saveTrainingArtifacts.value

    def setSaveTrainingArtifactsEnabled(self, enabled: bool) -> None:
        self._settings.saveTrainingArtifacts.value = enabled

    def getOutputPath(self) -> Path:
        return self._settings.outputPath.value

    def setOutputPath(self, directory: Path) -> None:
        self._settings.outputPath.value = directory

    def getOutputSuffix(self) -> str:
        return self._settings.outputSuffix.value

    def setOutputSuffix(self, suffix: str) -> None:
        self._settings.outputSuffix.value = suffix

    def getStatusIntervalInEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getStatusIntervalInEpochs(self) -> int:
        limits = self.getStatusIntervalInEpochsLimits()
        return limits.clamp(self._settings.statusIntervalInEpochs.value)

    def setStatusIntervalInEpochs(self, value: int) -> None:
        self._settings.statusIntervalInEpochs.value = value

    def getTrainingDataFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getTrainingDataFileFilter(self) -> str:
        return self._fileFilterList[0]

    def _createAxis(self, ticks: int, tickSize: float,
                    center: float) -> numpy.typing.NDArray[numpy.float_]:
        axis = numpy.arange(ticks) * tickSize
        axis += center - axis.mean()
        return axis

    def getPhaseCenteringStrategyList(self) -> Sequence[str]:
        return self._phaseCenteringStrategyChooser.getDisplayNameList()

    def getPhaseCenteringStrategy(self) -> str:
        return self._phaseCenteringStrategyChooser.getCurrentDisplayName()

    def setPhaseCenteringStrategy(self, name: str) -> None:
        self._phaseCenteringStrategyChooser.setFromDisplayName(name)

    def _syncFromSettings(self) -> None:
        self._phaseCenteringStrategyChooser.setFromSimpleName(
            self._settings.phaseCenteringStrategy.value)

    def _syncToSettings(self) -> None:
        self._settings.phaseCenteringStrategy.value = \
                self._phaseCenteringStrategyChooser.getCurrentSimpleName()

    def saveTrainingData(self, filePath: Path) -> None:
        selectedScan = self._scanAPI.getSelectedScan()
        selectedProbeArray = self._probeAPI.getSelectedProbeArray()
        selectedObjectArray = self._objectAPI.getSelectedObjectArray()

        if selectedScan is None:
            raise ValueError('No scan is selected!')

        if selectedProbeArray is None:
            raise ValueError('No probe is selected!')

        if selectedObjectArray is None:
            raise ValueError('No object is selected!')

        # FIXME objectPhaseCenteringStrategy = \
        # FIXME         self._phaseCenteringStrategyChooser.getCurrentStrategy()
        # FIXME object_ = objectPhaseCenteringStrategy(selectedObjectArray)

        diffractionData = self._diffractionDataset.getAssembledData()
        patches = numpy.zeros_like(diffractionData, dtype=complex)
        scanPositionsXInMeters: list[float] = list()
        scanPositionsYInMeters: list[float] = list()

        for index in self._diffractionDataset.getAssembledIndexes():
            try:
                point = selectedScan[index]
            except KeyError:
                continue

            scanPositionsXInMeters.append(float(point.x))
            scanPositionsYInMeters.append(float(point.y))
            patches[index, ...] = self._objectAPI.getObjectPatch(point)

        scanPositionsInMeters = numpy.column_stack(
            (scanPositionsYInMeters, scanPositionsXInMeters))

        pixelSizeXInMeters = float(self._objectAPI.getPixelSizeXInMeters())
        pixelSizeYInMeters = float(self._objectAPI.getPixelSizeYInMeters())

        # FIXME redo for ptychodus
        data: dict[str, numpy.typing.NDArray[numpy.number[Any]]] = {
            'real': patches,
            'reciprocal': diffractionData,
            'position': scanPositionsInMeters,
            'probe': selectedProbeArray,
            'pixelsize': numpy.array([pixelSizeYInMeters, pixelSizeXInMeters]),
        }

        numpy.savez_compressed(filePath, **data)

    def train(self, trainingDirPath: Path) -> None:
        logger.debug(f'Train using data in {trainingDirPath}.')
        diffractionPatternsList = list()
        reconstructedPatchesList = list()
        trainingFilePathGlob: Generator[Path, None, None] = trainingDirPath.glob('*.npz')

        for trainingFilePath in trainingFilePathGlob:
            logger.debug(f'Reading training data from \"{trainingFilePath}\"...')

            try:
                with numpy.load(trainingFilePath) as trainingFile:
                    diffractionPatternsList.append(trainingFile['reciprocal'])
                    reconstructedPatchesList.append(trainingFile['real'])
            except Exception:
                logger.exception('Failed to load training data.')

        diffractionPatterns = numpy.concatenate(diffractionPatternsList, axis=0)
        reconstructedPatches = numpy.concatenate(reconstructedPatchesList, axis=0)

        if self._trainer is None:
            logger.error('Trainable reconstructor not found!')
        else:
            # NOTE PtychoNN writes training outputs using internal mechanisms
            self._trainer.train(diffractionPatterns, reconstructedPatches)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()
            self.notifyObservers()
        elif observable is self._phaseCenteringStrategyChooser:
            self._syncToSettings()


class PtychoNNReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, settingsRegistry: SettingsRegistry,
                 phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                 diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                 probeAPI: ProbeAPI, objectAPI: ObjectAPI) -> None:
        super().__init__()
        self._settings = PtychoNNModelSettings.createInstance(settingsRegistry)
        self._trainingSettings = PtychoNNTrainingSettings.createInstance(settingsRegistry)
        self.modelPresenter = PtychoNNModelPresenter.createInstance(self._settings)
        self.trainingPresenter = PtychoNNTrainingPresenter.createInstance(
            self._trainingSettings, phaseCenteringStrategyChooser, diffractionDataset, scanAPI,
            probeAPI, objectAPI, settingsRegistry)
        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                       diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                       probeAPI: ProbeAPI, objectAPI: ObjectAPI,
                       isDeveloperModeEnabled: bool) -> PtychoNNReconstructorLibrary:
        core = cls(settingsRegistry, phaseCenteringStrategyChooser, diffractionDataset, scanAPI,
                   probeAPI, objectAPI)

        try:
            from .reconstructor import PtychoNNPhaseOnlyReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('PhaseOnly'))
        else:
            trainableReconstructor = PtychoNNPhaseOnlyReconstructor(core._settings,
                                                                    core._trainingSettings,
                                                                    scanAPI, probeAPI, objectAPI,
                                                                    diffractionDataset)
            core.trainingPresenter.setTrainer(trainableReconstructor)
            core.reconstructorList.append(trainableReconstructor)

        return core

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructorList)
