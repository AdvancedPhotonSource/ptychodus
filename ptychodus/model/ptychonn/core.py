from __future__ import annotations
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Final
import logging

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import (NullReconstructor, Reconstructor, ReconstructorLibrary,
                                         TrainableReconstructor)
from ptychodus.api.settings import SettingsRegistry

from .model import PtychoNNModelProvider
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNModelPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoNNModelSettings) -> None:
        super().__init__()
        self._settings = settings

        settings.addObserver(self)

    def getNumberOfConvolutionKernelsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfConvolutionKernels(self) -> int:
        limits = self.getNumberOfConvolutionKernelsLimits()
        return limits.clamp(self._settings.numberOfConvolutionKernels.value)

    def setNumberOfConvolutionKernels(self, value: int) -> None:
        self._settings.numberOfConvolutionKernels.value = value

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

    def __init__(self, settings: PtychoNNTrainingSettings) -> None:
        super().__init__()
        self._settings = settings

        settings.addObserver(self)

    def getValidationSetFractionalSizeLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getValidationSetFractionalSize(self) -> Decimal:
        limits = self.getValidationSetFractionalSizeLimits()
        return limits.clamp(self._settings.validationSetFractionalSize.value)

    def setValidationSetFractionalSize(self, value: Decimal) -> None:
        self._settings.validationSetFractionalSize.value = value

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

    def getStatusIntervalInEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getStatusIntervalInEpochs(self) -> int:
        limits = self.getStatusIntervalInEpochsLimits()
        return limits.clamp(self._settings.statusIntervalInEpochs.value)

    def setStatusIntervalInEpochs(self, value: int) -> None:
        self._settings.statusIntervalInEpochs.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoNNReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, modelSettings: PtychoNNModelSettings,
                 trainingSettings: PtychoNNTrainingSettings,
                 reconstructors: Sequence[Reconstructor]) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self.modelPresenter = PtychoNNModelPresenter(modelSettings)
        self.trainingPresenter = PtychoNNTrainingPresenter(trainingSettings)
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool) -> PtychoNNReconstructorLibrary:
        modelSettings = PtychoNNModelSettings(settingsRegistry)
        trainingSettings = PtychoNNTrainingSettings(settingsRegistry)
        phaseOnlyModelProvider = PtychoNNModelProvider(modelSettings,
                                                       trainingSettings,
                                                       enableAmplitude=False)
        amplitudePhaseModelProvider = PtychoNNModelProvider(modelSettings,
                                                            trainingSettings,
                                                            enableAmplitude=True)
        phaseOnlyReconstructor: TrainableReconstructor = NullReconstructor('PhaseOnly')
        amplitudePhaseReconstructor: TrainableReconstructor = NullReconstructor('AmplitudePhase')
        reconstructors: list[TrainableReconstructor] = list()

        try:
            from .reconstructor import PtychoNNTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if isDeveloperModeEnabled:
                reconstructors.append(phaseOnlyReconstructor)
                reconstructors.append(amplitudePhaseReconstructor)
        else:
            phaseOnlyReconstructor = PtychoNNTrainableReconstructor(modelSettings,
                                                                    trainingSettings,
                                                                    phaseOnlyModelProvider)
            amplitudePhaseReconstructor = PtychoNNTrainableReconstructor(
                modelSettings, trainingSettings, amplitudePhaseModelProvider)
            reconstructors.append(phaseOnlyReconstructor)
            reconstructors.append(amplitudePhaseReconstructor)

        return cls(modelSettings, trainingSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
