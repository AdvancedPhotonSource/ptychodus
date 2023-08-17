from __future__ import annotations
from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Final
import logging

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.reconstructor import (NullReconstructor, Reconstructor, ReconstructorLibrary,
                                  TrainableReconstructor)
from ...api.settings import SettingsRegistry
from ..object import ObjectAPI
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNModelPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, modelSettings: PtychoNNModelSettings) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._fileFilterList: list[str] = ['PyTorch Model State Files (*.pt *.pth)']

    @classmethod
    def createInstance(cls, modelSettings: PtychoNNModelSettings) -> PtychoNNModelPresenter:
        presenter = cls(modelSettings)
        modelSettings.addObserver(presenter)
        return presenter

    def getStateFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getStateFileFilter(self) -> str:
        return self._fileFilterList[0]

    def getStateFilePath(self) -> Path:
        return self._modelSettings.stateFilePath.value

    def setStateFilePath(self, directory: Path) -> None:
        self._modelSettings.stateFilePath.value = directory

    def getNumberOfConvolutionChannelsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfConvolutionChannels(self) -> int:
        limits = self.getNumberOfConvolutionChannelsLimits()
        return limits.clamp(self._modelSettings.numberOfConvolutionChannels.value)

    def setNumberOfConvolutionChannels(self, value: int) -> None:
        self._modelSettings.numberOfConvolutionChannels.value = value

    def getBatchSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getBatchSize(self) -> int:
        limits = self.getBatchSizeLimits()
        return limits.clamp(self._modelSettings.batchSize.value)

    def setBatchSize(self, value: int) -> None:
        self._modelSettings.batchSize.value = value

    def isBatchNormalizationEnabled(self) -> bool:
        return self._modelSettings.useBatchNormalization.value

    def setBatchNormalizationEnabled(self, enabled: bool) -> None:
        self._modelSettings.useBatchNormalization.value = enabled

    def update(self, observable: Observable) -> None:
        if observable is self._modelSettings:
            self.notifyObservers()


class PtychoNNTrainingPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, modelSettings: PtychoNNTrainingSettings,
                 trainer: TrainableReconstructor) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainer = trainer

    @classmethod
    def createInstance(cls, modelSettings: PtychoNNTrainingSettings,
                       trainer: TrainableReconstructor) -> PtychoNNTrainingPresenter:
        presenter = cls(modelSettings, trainer)
        modelSettings.addObserver(presenter)
        return presenter

    def getValidationSetFractionalSizeLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getValidationSetFractionalSize(self) -> Decimal:
        limits = self.getValidationSetFractionalSizeLimits()
        return limits.clamp(self._modelSettings.validationSetFractionalSize.value)

    def setValidationSetFractionalSize(self, value: Decimal) -> None:
        self._modelSettings.validationSetFractionalSize.value = value

    def getOptimizationEpochsPerHalfCycleLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getOptimizationEpochsPerHalfCycle(self) -> int:
        limits = self.getOptimizationEpochsPerHalfCycleLimits()
        return limits.clamp(self._modelSettings.optimizationEpochsPerHalfCycle.value)

    def setOptimizationEpochsPerHalfCycle(self, value: int) -> None:
        self._modelSettings.optimizationEpochsPerHalfCycle.value = value

    def getMaximumLearningRateLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getMaximumLearningRate(self) -> Decimal:
        limits = self.getMaximumLearningRateLimits()
        return limits.clamp(self._modelSettings.maximumLearningRate.value)

    def setMaximumLearningRate(self, value: Decimal) -> None:
        self._modelSettings.maximumLearningRate.value = value

    def getMinimumLearningRateLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getMinimumLearningRate(self) -> Decimal:
        limits = self.getMinimumLearningRateLimits()
        return limits.clamp(self._modelSettings.minimumLearningRate.value)

    def setMinimumLearningRate(self, value: Decimal) -> None:
        self._modelSettings.minimumLearningRate.value = value

    def getTrainingEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getTrainingEpochs(self) -> int:
        limits = self.getTrainingEpochsLimits()
        return limits.clamp(self._modelSettings.trainingEpochs.value)

    def setTrainingEpochs(self, value: int) -> None:
        self._modelSettings.trainingEpochs.value = value

    def isSaveTrainingArtifactsEnabled(self) -> bool:
        return self._modelSettings.saveTrainingArtifacts.value

    def setSaveTrainingArtifactsEnabled(self, enabled: bool) -> None:
        self._modelSettings.saveTrainingArtifacts.value = enabled

    def getOutputPath(self) -> Path:
        return self._modelSettings.outputPath.value

    def setOutputPath(self, directory: Path) -> None:
        self._modelSettings.outputPath.value = directory

    def getOutputSuffix(self) -> str:
        return self._modelSettings.outputSuffix.value

    def setOutputSuffix(self, suffix: str) -> None:
        self._modelSettings.outputSuffix.value = suffix

    def getStatusIntervalInEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getStatusIntervalInEpochs(self) -> int:
        limits = self.getStatusIntervalInEpochsLimits()
        return limits.clamp(self._modelSettings.statusIntervalInEpochs.value)

    def setStatusIntervalInEpochs(self, value: int) -> None:
        self._modelSettings.statusIntervalInEpochs.value = value

    def train(self) -> None:
        self._trainer.train()

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def saveTrainingData(self, filePath: Path) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"NPZ\"')
        self._trainer.saveTrainingData(filePath)

    def update(self, observable: Observable) -> None:
        if observable is self._modelSettings:
            self.notifyObservers()


class PtychoNNReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, modelSettings: PtychoNNModelSettings,
                 trainingSettings: PtychoNNTrainingSettings,
                 phaseOnlyTrainableReconstructor: TrainableReconstructor,
                 reconstructors: Sequence[Reconstructor]) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self.modelPresenter = PtychoNNModelPresenter.createInstance(modelSettings)
        self.trainingPresenter = PtychoNNTrainingPresenter.createInstance(
            trainingSettings, phaseOnlyTrainableReconstructor)
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry, objectAPI: ObjectAPI,
                       isDeveloperModeEnabled: bool) -> PtychoNNReconstructorLibrary:
        modelSettings = PtychoNNModelSettings.createInstance(settingsRegistry)
        trainingSettings = PtychoNNTrainingSettings.createInstance(settingsRegistry)
        phaseOnlyTrainableReconstructor: TrainableReconstructor = NullReconstructor('PhaseOnly')
        reconstructors: list[TrainableReconstructor] = list()

        try:
            from .phaseOnly import PtychoNNPhaseOnlyTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if isDeveloperModeEnabled:
                reconstructors.append(phaseOnlyTrainableReconstructor)
        else:
            phaseOnlyTrainableReconstructor = PtychoNNPhaseOnlyTrainableReconstructor(
                modelSettings, trainingSettings, objectAPI)
            reconstructors.append(phaseOnlyTrainableReconstructor)

        return cls(modelSettings, trainingSettings, phaseOnlyTrainableReconstructor,
                   reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
