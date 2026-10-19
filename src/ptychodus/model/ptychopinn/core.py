from __future__ import annotations
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import logging
from typing import Final

from ...api.reconstruct import (
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ...api.settings import SettingsRegistry
from .enums import PtychoPINNEnumerators
from .settings import (
    PtychoPINNDataSettings,
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

logger = logging.getLogger(__name__)

# Reconstructor names, also the backend's ``mode`` values. Shared so the
# developer-mode placeholders advertise the same names as the real thing.
_MODES: Final[tuple[str, ...]] = ('Unsupervised', 'Supervised')


def _ptychopinn_available() -> bool:
    """Return True iff ptycho_torch and lightning are importable, without importing them."""
    return all(find_spec(mod) is not None for mod in ('ptycho_torch', 'lightning'))


class PtychoPINNReconstructorLibrary(ReconstructorLibrary):
    def __init__(self, settings_registry: SettingsRegistry) -> None:
        super().__init__('ptychopinn')
        self.data_settings = PtychoPINNDataSettings(settings_registry)
        self.model_settings = PtychoPINNModelSettings(settings_registry)
        self.training_settings = PtychoPINNTrainingSettings(settings_registry)
        self.inference_settings = PtychoPINNInferenceSettings(settings_registry)
        self.enumerators = PtychoPINNEnumerators()
        self._reconstructors: list[TrainableReconstructor] = list()

        if not _ptychopinn_available():
            logger.info('PtychoPINN not found.')
            return

        # find_spec succeeded above, but the metadata query still hits importlib
        # metadata (not the module itself), so no torch/lightning import happens.
        try:
            ptychopinn_version = version('ptychopinn')
        except PackageNotFoundError:
            ptychopinn_version = 'unknown'
        logger.info(f'PtychoPINN {ptychopinn_version}')

        # Import the parent-side factory lazily so that this module's import
        # cost stays small even in headless mode.
        from .reconstructor import build_reconstructor

        for mode in _MODES:
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
        return 'PtychoPINN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
