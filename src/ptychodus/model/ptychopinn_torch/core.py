from __future__ import annotations
from collections.abc import Iterator
from importlib.metadata import version
import logging

from ...api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ...api.settings import SettingsRegistry
from .settings import (
    PtychoPINNTorchInferenceSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
)

logger = logging.getLogger(__name__)


class PtychoPINNTorchReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('ptychopinn-torch')
        self.model_settings = PtychoPINNTorchModelSettings(settings_registry)
        self.training_settings = PtychoPINNTorchTrainingSettings(settings_registry)
        self.inference_settings = PtychoPINNTorchInferenceSettings(settings_registry)
        self._reconstructors: list[TrainableReconstructor] = list()

        try:
            from .reconstructor import PtychoPINNTorchTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoPINN-Torch not found.')

            if is_developer_mode_enabled:
                for reconstructor in ('PINN', 'Supervised'):
                    self._reconstructors.append(NullReconstructor(reconstructor))
        else:
            ptychopinn_torch_version = version('ptychopinn_torch')
            logger.info(f'PtychoPINN-Torch {ptychopinn_torch_version}')

            self._reconstructors.append(
                PtychoPINNTorchTrainableReconstructor(
                    'Unsupervised',
                    self.model_settings,
                    self.inference_settings,
                    self.training_settings,
                    is_developer_mode_enabled=is_developer_mode_enabled,
                )
            )
            self._reconstructors.append(
                PtychoPINNTorchTrainableReconstructor(
                    'Supervised',
                    self.model_settings,
                    self.inference_settings,
                    self.training_settings,
                    is_developer_mode_enabled=is_developer_mode_enabled,
                )
            )

    @property
    def name(self) -> str:
        return 'PtychoPINN-Torch'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
