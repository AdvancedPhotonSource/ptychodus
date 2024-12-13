from __future__ import annotations
from collections.abc import Iterator
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
                self._reconstructors.append(NullReconstructor('PINN'))
                self._reconstructors.append(NullReconstructor('Supervised'))
        else:
            self._reconstructors.append(
                PtychoPINNTrainableReconstructor(
                    'PINN', self.model_settings, self.training_settings, self.inference_settings
                )
            )
            self._reconstructors.append(
                PtychoPINNTrainableReconstructor(
                    'Supervised',
                    self.model_settings,
                    self.training_settings,
                    self.inference_settings,
                )
            )

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    @property
    def logger_name(self) -> str:
        return 'ptychopinn'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
