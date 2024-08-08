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

from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings, PtychoNNPositionPredictionSettings
from ...model.ptychonn.position import PositionPredictionWorker

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

class PtychoNNPositionPredictionPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoNNPositionPredictionSettings) -> None:
        super().__init__()
        self._settings = settings
        self._reconImageFileFilterList: list[str] = ['TIFF images (*.tif *.tiff)']
        self._probePositionFileFilterList: list[str] = ['Probe position files (*.csv)']
        self._worker = PositionPredictionWorker(settings)
        settings.addObserver(self)

    def getReconstructedImageFileFilterList(self) -> Sequence[str]:
        return self._reconImageFileFilterList

    def getReconstructorImageFileFilter(self) -> str:
        return self._reconImageFileFilterList[0]
    
    def getProbePositionFileFilterList(self) -> Sequence[str]:
        return self._probePositionFileFilterList
    
    def getReconstructorImageFilePath(self) -> Path:
        return self._settings.reconstructorImagePath.value
    
    def setReconstructedImageFilePath(self, directory: Path) -> None:
        self._settings.reconstructorImagePath.value = directory

    def getProbePositionListFilePath(self) -> str:
        return self._settings.probePositionListPath.value
    
    def setProbePositionListFilePath(self, directory: Path) -> str:
        self._settings.probePositionListPath.value = directory
        
    def getProbePositionDataUnit(self) -> str:
        return self._settings.probePositionDataUnit.value
    
    def setProbePositionDataUnit(self, unit: str) -> None:
        self._settings.probePositionDataUnit.value = unit
        
    def getPixelSizeNM(self) -> Decimal:
        return self._settings.pixelSizeNM.value
    
    def setPixelSizeNM(self, value: Decimal) -> None:
        self._settings.pixelSizeNM.value = value
        
    def getBaselinePositionListFilePath(self) -> str:
        return self._settings.baselinePositionListPath.value
    
    def setBaselinePositionListFilePath(self, directory: Path) -> str:
        self._settings.baselinePositionListPath.value = directory
        
    def getCentralCrop(self) -> str:
        return self._settings.centralCrop.value
    
    def setCentralCrop(self, value: str) -> None:
        self._settings.centralCrop.value = value
        
    def getMethod(self) -> str:
        return self._settings.method.value
    
    def setMethod(self, value: str) -> None:
        self._settings.method.value = value

    def getNumberNeighborsCollectiveLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberNeighborsCollective(self) -> int:
        limits = self.getNumberNeighborsCollectiveLimits()
        return limits.clamp(self._settings.numberNeighborsCollective.value)

    def setNumberNeighborsCollective(self, value: int) -> None:
        self._settings.numberNeighborsCollective.value = value
        
    def getOffsetEstimatorOrder(self) -> int:
        return self._settings.offsetEstimatorOrder.value
    
    def setOffsetEstimatorOrder(self, value: int) -> None:
        self._settings.offsetEstimatorOrder.value = value
        
    def getOffsetEstimatorBeta(self) -> Decimal:
        return self._settings.offsetEstimatorBeta.value
    
    def setOffsetEstimatorBeta(self, value: Decimal) -> None:
        self._settings.offsetEstimatorBeta.value = value
        
    def getSmoothConstraintWeight(self) -> Decimal:
        return self._settings.smoothConstraintWeight.value
    
    def setSmoothConstraintWeight(self, value: Decimal) -> None:
        self._settings.smoothConstraintWeight.value = value
        
    def getRectangularGrid(self) -> bool:
        return self._settings.rectangularGrid.value
    
    def setRectangularGrid(self, value: bool) -> None:
        self._settings.rectangularGrid.value = value
        
    def getRandomSeed(self) -> int:
        return self._settings.randomSeed.value
    
    def setRandomSeed(self, value: int) -> None:
        self._settings.randomSeed.value = value
        
    def getDebug(self) -> bool:
        return self._settings.debug.value
    
    def setDebug(self, value: bool) -> None:
        self._settings.debug.value = value
        
    def getRegistrationMethod(self) -> str:
        return self._settings.registrationMethod.value
    
    def setRegistrationMethod(self, value: str) -> None:
        self._settings.registrationMethod.value = value
        
    def getHybridRegistrationTols(self) -> str:
        return self._settings.hybridRegistrationTols.value
    
    def setHybridRegistrationTols(self, value: str) -> None:
        self._settings.hybridRegistrationTols.value = value
        
    def getNonHybridRegistrationTol(self) -> Decimal:
        return self._settings.nonHybridRegistrationTol.value
    
    def setNonHybridRegistrationTol(self, value: Decimal) -> None:
        self._settings.nonHybridRegistrationTol.value = value
        
    def getMaxShift(self) -> int:
        return self._settings.maxShift.value
    
    def setMaxShift(self, value: int) -> None:
        self._settings.maxShift.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()

    def runPositionPrediction(self):
        self._worker.run()

class PtychoNNReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, modelSettings: PtychoNNModelSettings,
                 trainingSettings: PtychoNNTrainingSettings,
                 positionPredictionSettings: PtychoNNPositionPredictionSettings,
                 reconstructors: Sequence[Reconstructor]) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._positionPredictionSettings = positionPredictionSettings
        self.modelPresenter = PtychoNNModelPresenter(modelSettings)
        self.trainingPresenter = PtychoNNTrainingPresenter(trainingSettings)
        self.positionPredictionPresenter = PtychoNNPositionPredictionPresenter(positionPredictionSettings)
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool) -> PtychoNNReconstructorLibrary:
        modelSettings = PtychoNNModelSettings(settingsRegistry)
        trainingSettings = PtychoNNTrainingSettings(settingsRegistry)
        positionPredictionSettings = PtychoNNPositionPredictionSettings(settingsRegistry)
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
            phaseOnlyModelProvider = PtychoNNModelProvider(modelSettings,
                                                           trainingSettings,
                                                           enableAmplitude=False)
            phaseOnlyReconstructor = PtychoNNTrainableReconstructor(modelSettings,
                                                                    trainingSettings,
                                                                    phaseOnlyModelProvider)
            amplitudePhaseModelProvider = PtychoNNModelProvider(modelSettings,
                                                                trainingSettings,
                                                                enableAmplitude=True)
            amplitudePhaseReconstructor = PtychoNNTrainableReconstructor(
                modelSettings, trainingSettings, amplitudePhaseModelProvider)
            reconstructors.append(phaseOnlyReconstructor)
            reconstructors.append(amplitudePhaseReconstructor)

        return cls(modelSettings, trainingSettings, positionPredictionSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
