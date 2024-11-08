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
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoPINNModelPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoPINNModelSettings) -> None:
        super().__init__()
        self._settings = settings
        self._fileFilterList: list[str] = ['PyTorch Model State Files (*.pt *.pth)']

    @classmethod
    def createInstance(cls, settings: PtychoPINNModelSettings) -> PtychoPINNModelPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getStateFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getStateFileFilter(self) -> str:
        return self._fileFilterList[0]

    def getGridsizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNFiltersScaleLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNPhotonsLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('1e0'), Decimal('1e12'))

    def getProbeScaleLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('1e-3'), Decimal('1e3'))

    def getSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def setStateFilePath(self, path: Path) -> None:
        self._stateFilePath = path
        self.notifyObservers()

    def getStateFilePath(self) -> Path:
        return self._stateFilePath

    def getGridsize(self) -> int:
        return self._settings.gridsize.value

    def setGridsize(self, value: int) -> None:
        if 1 <= value <= self.MAX_INT:
            self._settings.gridsize.value = value
            self.notifyObservers()

    def getBatchSize(self) -> int:
        return self._settings.batchSize.value

    def setBatchSize(self, value: int) -> None:
        self._settings.batchSize.value = value

    def getNFiltersScale(self) -> int:
        return self._settings.nFiltersScale.value

    def setNFiltersScale(self, value: int) -> None:
        self._settings.nFiltersScale.value = value

    def isProbeTrainable(self) -> bool:
        return self._settings.probeTrainable.value

    def setProbeTrainable(self, enabled: bool) -> None:
        self._settings.probeTrainable.value = enabled

    def isIntensityScaleTrainable(self) -> bool:
        return self._settings.intensityScaleTrainable.value

    def setIntensityScaleTrainable(self, enabled: bool) -> None:
        self._settings.intensityScaleTrainable.value = enabled

    def isObjectBig(self) -> bool:
        return self._settings.objectBig.value

    def setObjectBig(self, enabled: bool) -> None:
        self._settings.objectBig.value = enabled

    def isProbeBig(self) -> bool:
        return self._settings.probeBig.value

    def setProbeBig(self, enabled: bool) -> None:
        self._settings.probeBig.value = enabled

    def getProbeScale(self) -> Decimal:
        return self._settings.probeScale.value

    def setProbeScale(self, value: Decimal) -> None:
        self._settings.probeScale.value = value

    def isProbeMask(self) -> bool:
        return self._settings.probeMask.value

    def setProbeMask(self, enabled: bool) -> None:
        self._settings.probeMask.value = enabled

    def getAmpActivation(self) -> str:
        return self._settings.ampActivation.value

    def setAmpActivation(self, ampActivation: str) -> None:
        self._settings.ampActivation.value = ampActivation

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoPINNTrainingPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoPINNTrainingSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: PtychoPINNTrainingSettings) -> PtychoPINNTrainingPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def setValidationSetFractionalSize(self, value: Decimal) -> None:
        self._settings.validationSetFractionalSize.value = value

    def setMaximumLearningRate(self, value: Decimal) -> None:
        self._settings.maximumLearningRate.value = value

    def setMinimumLearningRate(self, value: Decimal) -> None:
        self._settings.minimumLearningRate.value = value

    def setTrainingEpochs(self, value: int) -> None:
        self._settings.trainingEpochs.value = value

    def setMaeWeight(self, value: Decimal) -> None:
        self._settings.maeWeight.value = value

    def setNllWeight(self, value: Decimal) -> None:
        self._settings.nllWeight.value = value

    def setRealspaceMAEWeight(self, value: Decimal) -> None:
        self._settings.realspaceMAEWeight.value = value

    def setRealspaceWeight(self, value: Decimal) -> None:
        self._settings.realspaceWeight.value = value

    def getValidationSetFractionalSize(self) -> Decimal:
        return self._settings.validationSetFractionalSize.value

    def getValidationSetFractionalSizeLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('0'), Decimal('1'))

    def getMaximumLearningRate(self) -> Decimal:
        return self._settings.maximumLearningRate.value

    def getMinimumLearningRate(self) -> Decimal:
        return self._settings.minimumLearningRate.value

    def getTrainingEpochs(self) -> int:
        return self._settings.trainingEpochs.value

    def getTrainingEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getMaeWeight(self) -> Decimal:
        return self._settings.maeWeight.value

    def getNllWeight(self) -> Decimal:
        return self._settings.nllWeight.value

    def getRealspaceMAEWeight(self) -> Decimal:
        return self._settings.realspaceMAEWeight.value

    def getRealspaceWeight(self) -> Decimal:
        return self._settings.realspaceWeight.value

    def getNEpochs(self) -> int:
        return self._settings.trainingEpochs.value

    def setNEpochs(self, value: int) -> None:
        self._settings.trainingEpochs.value = value

    def getOutputPath(self) -> Path:
        return self._settings.outputPath.value

    def setOutputPath(self, directory: Path) -> None:
        self._settings.outputPath.value = directory

    def getOutputSuffix(self) -> str:
        return self._settings.outputSuffix.value

    def setOutputSuffix(self, suffix: str) -> None:
        self._settings.outputSuffix.value = suffix

    def isSaveTrainingArtifactsEnabled(self) -> bool:
        return self._settings.saveTrainingArtifacts.value

    def setSaveTrainingArtifactsEnabled(self, enabled: bool) -> None:
        self._settings.saveTrainingArtifacts.value = enabled

    def getMAEWeightLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('0'), Decimal('1'))

    def getNLLWeightLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('0'), Decimal('1'))

    def getTVWeightLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('0'), Decimal('1'))

    def getRealspaceMAEWeightLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('0'), Decimal('1'))

    def getRealspaceWeightLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal('0'), Decimal('1'))

    def getEpochsLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getEpochs(self) -> int:
        limits = self.getEpochsLimits()
        return limits.clamp(self._settings.trainingEpochs.value)

    def setEpochs(self, value: int) -> None:
        self._settings.trainingEpochs.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoPINNReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, modelSettings: PtychoPINNModelSettings,
                 trainingSettings: PtychoPINNTrainingSettings,
                 reconstructors: Sequence[Reconstructor]) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self.modelPresenter = PtychoPINNModelPresenter.createInstance(modelSettings)
        self.trainingPresenter = PtychoPINNTrainingPresenter.createInstance(trainingSettings)
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool) -> PtychoPINNReconstructorLibrary:
        modelSettings = PtychoPINNModelSettings(settingsRegistry)
        trainingSettings = PtychoPINNTrainingSettings(settingsRegistry)
        ptychoPINNReconstructor: TrainableReconstructor = NullReconstructor('PtychoPINN')
        reconstructors: list[TrainableReconstructor] = list()

        try:
            from .reconstructor import PtychoPINNTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoPINN not found.')

            if isDeveloperModeEnabled:
                reconstructors.append(ptychoPINNReconstructor)
        else:
            ptychoPINNReconstructor = PtychoPINNTrainableReconstructor(
                modelSettings, trainingSettings)
            reconstructors.append(ptychoPINNReconstructor)

        return cls(modelSettings, trainingSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
