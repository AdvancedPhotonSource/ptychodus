from __future__ import annotations
from collections.abc import Iterator, Sequence
import logging

from ...api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ...api.settings import SettingsRegistry
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoPINNPresenter:
    def __init__(self) -> None:
        self._fileFilterList: list[str] = ['PyTorch Model State Files (*.pt *.pth)']

    def getStateFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getStateFileFilter(self) -> str:
        return self._fileFilterList[0]


class PtychoPINNReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self,
        modelSettings: PtychoPINNModelSettings,
        trainingSettings: PtychoPINNTrainingSettings,
        reconstructors: Sequence[Reconstructor],
    ) -> None:
        super().__init__()
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self.presenter = PtychoPINNPresenter()
        self._reconstructors = reconstructors

    @classmethod
    def createInstance(
        cls, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool
    ) -> PtychoPINNReconstructorLibrary:
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
                modelSettings, trainingSettings
            )
            reconstructors.append(ptychoPINNReconstructor)

        return cls(modelSettings, trainingSettings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
