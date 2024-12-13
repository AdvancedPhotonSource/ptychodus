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
from .enums import PtychoPINNEnumerators
from .settings import (
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

logger = logging.getLogger(__name__)


class PtychoPINNPresenter:
    def __init__(self) -> None:
        self._fileFilterList: list[str] = ['PyTorch Model State Files (*.pt *.pth)']

    def getStateFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getStateFileFilter(self) -> str:
        return self._fileFilterList[0]


class PtychoPINNReconstructorLibrary(ReconstructorLibrary):
    def __init__(self, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool) -> None:
        super().__init__()
        self.model_settings = PtychoPINNModelSettings(settingsRegistry)
        self.training_settings = PtychoPINNTrainingSettings(settingsRegistry)
        self.inference_settings = PtychoPINNInferenceSettings(settingsRegistry)
        self.enumerators = PtychoPINNEnumerators()
        self._reconstructors: list[TrainableReconstructor] = list()

        try:
            from .reconstructor import PtychoPINNTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoPINN not found.')

            if isDeveloperModeEnabled:
                reconstructor: TrainableReconstructor = NullReconstructor('PtychoPINN')
                self._reconstructors.append(reconstructor)
        else:
            reconstructor = PtychoPINNTrainableReconstructor(
                self.model_settings, self.training_settings, self.inference_settings
            )
            self._reconstructors.append(reconstructor)

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
