from __future__ import annotations
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import logging

from ...api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)
from ...api.settings import SettingsRegistry
from .enums import PtychoPINNTorchEnumerators
from .settings import (
    PtychoPINNTorchDataSettings,
    PtychoPINNTorchInferenceSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
)

logger = logging.getLogger(__name__)


def _ptychopinn_torch_available() -> bool:
    """Return True iff ptycho_torch and lightning are importable, without importing them."""
    return all(find_spec(mod) is not None for mod in ('ptycho_torch', 'lightning'))


class PtychoPINNTorchReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> None:
        super().__init__('ptychopinn-torch')
        self.data_settings = PtychoPINNTorchDataSettings(settings_registry)
        self.model_settings = PtychoPINNTorchModelSettings(settings_registry)
        self.training_settings = PtychoPINNTorchTrainingSettings(settings_registry)
        self.inference_settings = PtychoPINNTorchInferenceSettings(settings_registry)
        self.enumerators = PtychoPINNTorchEnumerators()
        self._reconstructors: list[TrainableReconstructor] = list()

        if not _ptychopinn_torch_available():
            logger.info('PtychoPINN-Torch not found.')

            if is_developer_mode_enabled:
                for reconstructor in ('PINN', 'Supervised'):
                    self._reconstructors.append(NullReconstructor(reconstructor))
            return

        # find_spec succeeded above, but the metadata query still hits importlib
        # metadata (not the module itself), so no torch/lightning import happens.
        try:
            ptychopinn_torch_version = version('ptychopinn')
        except PackageNotFoundError:
            ptychopinn_torch_version = 'unknown'
        logger.info(f'PtychoPINN-Torch {ptychopinn_torch_version}')

        # Import the parent-side factory lazily so that this module's import
        # cost stays small even in headless mode.
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
        return 'PtychoPINN-Torch'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self._reconstructors)
