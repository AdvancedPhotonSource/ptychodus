from __future__ import annotations
from collections.abc import Iterator
import logging

from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ptychodus.api.settings import SettingsRegistry

from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('ptychonn')
        self.model_settings = PtychoNNModelSettings(settings_registry)
        self.training_settings = PtychoNNTrainingSettings(settings_registry)
        self._reconstructors: list[TrainableReconstructor] = list()

        try:
            from .model import PtychoNNModelProvider
            from .reconstructor import PtychoNNTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if is_developer_mode_enabled:
                self._reconstructors.append(NullReconstructor('PhaseOnly'))
                self._reconstructors.append(NullReconstructor('AmplitudePhase'))
        else:
            phase_only_model_provider = PtychoNNModelProvider(
                self.model_settings, self.training_settings, enable_amplitude=False
            )
            amplitude_phase_model_provider = PtychoNNModelProvider(
                self.model_settings, self.training_settings, enable_amplitude=True
            )

            self._reconstructors.append(
                PtychoNNTrainableReconstructor(
                    self.model_settings, self.training_settings, phase_only_model_provider
                )
            )
            self._reconstructors.append(
                PtychoNNTrainableReconstructor(
                    self.model_settings, self.training_settings, amplitude_phase_model_provider
                )
            )

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
