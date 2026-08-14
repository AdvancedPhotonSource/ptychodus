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

from .enums import PtychoPINNEnumerators
from .settings import (
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

logger = logging.getLogger(__name__)


def _ptychopinn_available() -> bool:
    """Return True iff the ``ptycho`` package is importable, without importing it."""
    return find_spec('ptycho') is not None


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

        if not _ptychopinn_available():
            logger.info('PtychoPINN not found.')

            if is_developer_mode_enabled:
                for reconstructor in ('PINN', 'Supervised'):
                    self._reconstructors.append(NullReconstructor(reconstructor))
            return

        try:
            ptychopinn_version = version('ptychopinn')
        except PackageNotFoundError:
            try:
                ptychopinn_version = version('ptycho')
            except PackageNotFoundError:
                ptychopinn_version = 'unknown'

        logger.info(f'PtychoPINN {ptychopinn_version}')

        from .reconstructor import build_reconstructor

        for mode in ('PINN', 'Supervised'):
            self._reconstructors.append(
                build_reconstructor(
                    mode,
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
