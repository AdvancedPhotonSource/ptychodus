from __future__ import annotations
from collections.abc import Iterator, Sequence
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
        self,
        model_settings: PtychoNNModelSettings,
        training_settings: PtychoNNTrainingSettings,
        reconstructors: Sequence[Reconstructor],
    ) -> None:
        super().__init__('ptychonn')
        self.model_settings = model_settings
        self.training_settings = training_settings
        self._reconstructors = reconstructors

    @classmethod
    def create_instance(
        cls, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> PtychoNNReconstructorLibrary:
        model_settings = PtychoNNModelSettings(settings_registry)
        training_settings = PtychoNNTrainingSettings(settings_registry)
        phase_only_reconstructor: TrainableReconstructor = NullReconstructor('PhaseOnly')
        amplitude_phase_reconstructor: TrainableReconstructor = NullReconstructor('AmplitudePhase')
        reconstructors: list[TrainableReconstructor] = list()

        try:
            from .model import PtychoNNModelProvider
            from .reconstructor import PtychoNNTrainableReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if is_developer_mode_enabled:
                reconstructors.append(phase_only_reconstructor)
                reconstructors.append(amplitude_phase_reconstructor)
        else:
            phase_only_model_provider = PtychoNNModelProvider(
                model_settings, training_settings, enable_amplitude=False
            )
            phase_only_reconstructor = PtychoNNTrainableReconstructor(
                model_settings, training_settings, phase_only_model_provider
            )
            amplitude_phase_model_provider = PtychoNNModelProvider(
                model_settings, training_settings, enable_amplitude=True
            )
            amplitude_phase_reconstructor = PtychoNNTrainableReconstructor(
                model_settings, training_settings, amplitude_phase_model_provider
            )
            reconstructors.append(phase_only_reconstructor)
            reconstructors.append(amplitude_phase_reconstructor)

        return cls(model_settings, training_settings, reconstructors)

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
