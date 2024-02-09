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

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()

    _fileFilterList: list[str] = ['PyTorch Model State Files (*.pt *.pth)']

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

    def getOffset(self) -> int:
        return self._settings.offset.value

    def setOffset(self, value: int) -> None:
        self._settings.offset.value = value

    def getGridsize(self) -> int:
        return self._settings.gridsize.value

    def setGridsize(self, value: int) -> None:
        self._settings.gridsize.value = value

    def getOuterOffsetTrain(self) -> int:
        return self._settings.outer_offset_train.value

    def setOuterOffsetTrain(self, value: int) -> None:
        self._settings.outer_offset_train.value = value

    def getOuterOffsetTest(self) -> int:
        return self._settings.outer_offset_test.value

    def setOuterOffsetTest(self, value: int) -> None:
        self._settings.outer_offset_test.value = value

    def getBatchSize(self) -> int:
        return self._settings.batch_size.value

    def setBatchSize(self, value: int) -> None:
        self._settings.batch_size.value = value

    def getNEpochs(self) -> int:
        return self._settings.nepochs.value

    def setNEpochs(self, value: int) -> None:
        self._settings.nepochs.value = value

    def getNFiltersScale(self) -> int:
        return self._settings.n_filters_scale.value

    def setNFiltersScale(self, value: int) -> None:
        self._settings.n_filters_scale.value = value

    def getNPhotons(self) -> Decimal:
        return self._settings.nphotons.value

    def setNPhotons(self, value: Decimal) -> None:
        self._settings.nphotons.value = value

    def isProbeTrainable(self) -> bool:
        return self._settings.probe_trainable.value

    def setProbeTrainable(self, enabled: bool) -> None:
        self._settings.probe_trainable.value = enabled

    def isIntensityScaleTrainable(self) -> bool:
        return self._settings.intensity_scale_trainable.value

    def setIntensityScaleTrainable(self, enabled: bool) -> None:
        self._settings.intensity_scale_trainable.value = enabled

    def isObjectBig(self) -> bool:
        return self._settings.object_big.value

    def setObjectBig(self, enabled: bool) -> None:
        self._settings.object_big.value = enabled

    def isProbeBig(self) -> bool:
        return self._settings.probe_big.value

    def setProbeBig(self, enabled: bool) -> None:
        self._settings.probe_big.value = enabled

    def getProbeScale(self) -> Decimal:
        return self._settings.probe_scale.value

    def setProbeScale(self, value: Decimal) -> None:
        self._settings.probe_scale.value = value

    def isProbeMask(self) -> bool:
        return self._settings.probe_mask.value

    def setProbeMask(self, enabled: bool) -> None:
        self._settings.probe_mask.value = enabled

    def getModelType(self) -> str:
        return self._settings.model_type.value

    def setModelType(self, model_type: str) -> None:
        self._settings.model_type.value = model_type

    def getSize(self) -> int:
        return self._settings.size.value

    def setSize(self, value: int) -> None:
        self._settings.size.value = value

    def getAmpActivation(self) -> str:
        return self._settings.amp_activation.value

    def setAmpActivation(self, amp_activation: str) -> None:
        self._settings.amp_activation.value = amp_activation



class PtychoPINNTrainingPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoPINNTrainingSettings) -> None:
        super().__init__()
        self._settings = settings


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

    @classmethod
    def createInstance(cls, settings: PtychoPINNTrainingSettings) -> PtychoPINNTrainingPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    # Methods to interact with training settings have been defined

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
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNReconstructorLibrary:
        modelSettings = PtychoPINNModelSettings.createInstance(settingsRegistry)
        trainingSettings = PtychoPINNTrainingSettings.createInstance(settingsRegistry)
        ptychoPINNReconstructor: TrainableReconstructor = NullReconstructor('PtychoPINN')
        reconstructors: list[TrainableReconstructor] = [ptychoPINNReconstructor]
        return cls(modelSettings, trainingSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
