from __future__ import annotations
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import logging

from ptychodus.api.reconstruct import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ptychodus.api.settings import SettingsRegistry

from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


def _ptychonn_available() -> bool:
    """Return True iff ptychonn and lightning are importable, without importing them."""
    return all(find_spec(mod) is not None for mod in ('ptychonn', 'lightning'))


class PtychoNNReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('ptychonn')
        self.model_settings = PtychoNNModelSettings(settings_registry)
        self.training_settings = PtychoNNTrainingSettings(settings_registry)
        self._reconstructors: list[TrainableReconstructor] = list()

        if not _ptychonn_available():
            logger.info('PtychoNN not found.')

            if is_developer_mode_enabled:
                self._reconstructors.append(NullReconstructor('PhaseOnly'))
                self._reconstructors.append(NullReconstructor('AmplitudePhase'))
            return

        try:
            ptychonn_version = version('ptychonn')
        except PackageNotFoundError:
            ptychonn_version = 'unknown'
        logger.info(f'PtychoNN {ptychonn_version}')

        from .reconstructor import build_reconstructor

        self._reconstructors.append(
            build_reconstructor(
                'PhaseOnly',
                enable_amplitude=False,
                model_settings=self.model_settings,
                training_settings=self.training_settings,
            )
        )
        self._reconstructors.append(
            build_reconstructor(
                'AmplitudePhase',
                enable_amplitude=True,
                model_settings=self.model_settings,
                training_settings=self.training_settings,
            )
        )

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
