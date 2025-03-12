from __future__ import annotations
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Final
import logging

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ptychodus.api.settings import SettingsRegistry

from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNModelPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoNNModelSettings) -> None:
        super().__init__()
        self._settings = settings

        settings.add_observer(self)

    def getNumberOfConvolutionKernelsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfConvolutionKernels(self) -> int:
        limits = self.getNumberOfConvolutionKernelsLimits()
        return limits.clamp(self._settings.numberOfConvolutionKernels.get_value())

    def setNumberOfConvolutionKernels(self, value: int) -> None:
        self._settings.numberOfConvolutionKernels.set_value(value)

    def getBatchSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getBatchSize(self) -> int:
        limits = self.getBatchSizeLimits()
        return limits.clamp(self._settings.batchSize.get_value())

    def setBatchSize(self, value: int) -> None:
        self._settings.batchSize.set_value(value)

    def isBatchNormalizationEnabled(self) -> bool:
        return self._settings.useBatchNormalization.get_value()

    def setBatchNormalizationEnabled(self, enabled: bool) -> None:
        self._settings.useBatchNormalization.set_value(enabled)

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notify_observers()


class PtychoNNTrainingPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoNNTrainingSettings) -> None:
        super().__init__()
        self._settings = settings

        settings.add_observer(self)

    def getValidationSetFractionalSizeLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getValidationSetFractionalSize(self) -> Decimal:
        limits = self.getValidationSetFractionalSizeLimits()
        return limits.clamp(
            Decimal.from_float(self._settings.validationSetFractionalSize.get_value())
        )

    def setValidationSetFractionalSize(self, value: Decimal) -> None:
        self._settings.validationSetFractionalSize.set_value(float(value))

    def getMaximumLearningRateLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getMaximumLearningRate(self) -> Decimal:
        limits = self.getMaximumLearningRateLimits()
        return limits.clamp(Decimal.from_float(self._settings.maximumLearningRate.get_value()))

    def setMaximumLearningRate(self, value: Decimal) -> None:
        self._settings.maximumLearningRate.set_value(float(value))

    def getMinimumLearningRateLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getMinimumLearningRate(self) -> Decimal:
        limits = self.getMinimumLearningRateLimits()
        return limits.clamp(Decimal.from_float(self._settings.minimumLearningRate.get_value()))

    def setMinimumLearningRate(self, value: Decimal) -> None:
        self._settings.minimumLearningRate.set_value(float(value))

    def getTrainingEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getTrainingEpochs(self) -> int:
        limits = self.getTrainingEpochsLimits()
        return limits.clamp(self._settings.trainingEpochs.get_value())

    def setTrainingEpochs(self, value: int) -> None:
        self._settings.trainingEpochs.set_value(value)

    def getStatusIntervalInEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getStatusIntervalInEpochs(self) -> int:
        limits = self.getStatusIntervalInEpochsLimits()
        return limits.clamp(self._settings.statusIntervalInEpochs.get_value())

    def setStatusIntervalInEpochs(self, value: int) -> None:
        self._settings.statusIntervalInEpochs.set_value(value)

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notify_observers()


class PtychoNNReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self,
        modelSettings: PtychoNNModelSettings,
        trainingSettings: PtychoNNTrainingSettings,
        reconstructors: Sequence[Reconstructor],
    ) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self.modelPresenter = PtychoNNModelPresenter(modelSettings)
        self.trainingPresenter = PtychoNNTrainingPresenter(trainingSettings)
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(
        cls, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool
    ) -> PtychoNNReconstructorLibrary:
        modelSettings = PtychoNNModelSettings(settingsRegistry)
        trainingSettings = PtychoNNTrainingSettings(settingsRegistry)
        phaseOnlyReconstructor: TrainableReconstructor = NullReconstructor('PhaseOnly')
        amplitudePhaseReconstructor: TrainableReconstructor = NullReconstructor('AmplitudePhase')
        reconstructors: list[TrainableReconstructor] = list()

        try:
            from .model import PtychoNNModelProvider
            from .reconstructor import PtychoNNTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if isDeveloperModeEnabled:
                reconstructors.append(phaseOnlyReconstructor)
                reconstructors.append(amplitudePhaseReconstructor)
        else:
            phaseOnlyModelProvider = PtychoNNModelProvider(
                modelSettings, trainingSettings, enableAmplitude=False
            )
            phaseOnlyReconstructor = PtychoNNTrainableReconstructor(
                modelSettings, trainingSettings, phaseOnlyModelProvider
            )
            amplitudePhaseModelProvider = PtychoNNModelProvider(
                modelSettings, trainingSettings, enableAmplitude=True
            )
            amplitudePhaseReconstructor = PtychoNNTrainableReconstructor(
                modelSettings, trainingSettings, amplitudePhaseModelProvider
            )
            reconstructors.append(phaseOnlyReconstructor)
            reconstructors.append(amplitudePhaseReconstructor)

        return cls(modelSettings, trainingSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoNN'

    @property
    def logger_name(self) -> str:
        return 'ptychonn'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
