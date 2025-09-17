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
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('ptychopinn')
        self.model_settings = PtychoPINNModelSettings(settings_registry)
        self.training_settings = PtychoPINNTrainingSettings(settings_registry)
        self.inference_settings = PtychoPINNInferenceSettings(settings_registry)
        self.enumerators = PtychoPINNEnumerators()
        self._reconstructors: list[TrainableReconstructor] = list()

        try:
            from .reconstructor import PtychoPINNTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoPINN not found.')

            if is_developer_mode_enabled:
                self._reconstructors.append(NullReconstructor('PINN'))
                self._reconstructors.append(NullReconstructor('Supervised'))
        else:
            self._reconstructors.append(
                PtychoPINNTrainableReconstructor(
                    'PINN',
                    self.model_settings,
                    self.inference_settings,
                    self.training_settings,
                    is_developer_mode_enabled=is_developer_mode_enabled,
                )
            )
            self._reconstructors.append(
                PtychoPINNTrainableReconstructor(
                    'Supervised',
                    self.model_settings,
                    self.inference_settings,
                    self.training_settings,
                    is_developer_mode_enabled=is_developer_mode_enabled,
                )
            )

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
