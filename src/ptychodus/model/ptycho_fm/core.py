from __future__ import annotations

import logging
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

from ...api.reconstruct import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ...api.settings import SettingsRegistry
from .enums import PtychoFMEnumerators
from .settings import (
    PtychoFMDataSettings,
    PtychoFMInferenceSettings,
    PtychoFMModelSettings,
    PtychoFMTrainingSettings,
)

logger = logging.getLogger(__name__)


def _ptycho_fm_available() -> bool:
    """Return True iff every package the child imports is present, without importing them.

    ``ptychi`` is in the list because the child borrows
    ``ptychi.image_proc.place_patches_fourier_shift`` to stitch predicted
    patches. Only the reconstruct path needs it, but a reconstructor that can
    train and not reconstruct is not worth advertising, so a missing pty-chi
    disables the backend here rather than failing inside a spawned child.
    """
    return all(find_spec(mod) is not None for mod in ('ptycho_fm', 'torch', 'ptychi'))


class PtychoFMReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('ptycho-fm')
        self.data_settings = PtychoFMDataSettings(settings_registry)
        self.model_settings = PtychoFMModelSettings(settings_registry)
        self.training_settings = PtychoFMTrainingSettings(settings_registry)
        self.inference_settings = PtychoFMInferenceSettings(settings_registry)
        self.enumerators = PtychoFMEnumerators()
        self._reconstructors: list[TrainableReconstructor] = list()

        if not _ptycho_fm_available():
            logger.info('PtychoFM (ptycho-fm) not found.')

            if is_developer_mode_enabled:
                for reconstructor in ('Unsupervised', 'Supervised'):
                    self._reconstructors.append(NullReconstructor(reconstructor))
            return

        try:
            ptycho_fm_version = version('ptycho-fm')
        except PackageNotFoundError:
            ptycho_fm_version = 'unknown'
        logger.info(f'PtychoFM (ptycho-fm) {ptycho_fm_version}')

        # Lazy import: keeps this module's cost small in headless mode and
        # ensures the parent never pulls torch in just because ptycho-fm is
        # installed.
        from .reconstructor import build_reconstructor

        for mode in ('Unsupervised', 'Supervised'):
            self._reconstructors.append(
                build_reconstructor(
                    mode,
                    self.data_settings,
                    self.model_settings,
                    self.inference_settings,
                    self.training_settings,
                )
            )

    @property
    def name(self) -> str:
        return 'PtychoFM'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
