from __future__ import annotations
from collections.abc import Iterator
from importlib.metadata import version
import logging

from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from .settings import (
    TikeMultigridSettings,
    TikeObjectCorrectionSettings,
    TikePositionCorrectionSettings,
    TikeProbeCorrectionSettings,
    TikeSettings,
)

logger = logging.getLogger(__name__)


class TikeReconstructorLibrary(ReconstructorLibrary):
    def __init__(self, settings_registry: SettingsRegistry) -> None:
        super().__init__('tike')
        self.settings = TikeSettings(settings_registry)
        self.multigrid_settings = TikeMultigridSettings(settings_registry)
        self.position_correction_settings = TikePositionCorrectionSettings(settings_registry)
        self.probe_correction_settings = TikeProbeCorrectionSettings(settings_registry)
        self.object_correction_settings = TikeObjectCorrectionSettings(settings_registry)

        self.reconstructor_list: list[Reconstructor] = list()

    @classmethod
    def create_instance(
        cls, settings_registry: SettingsRegistry, is_developer_mode_enabled: bool
    ) -> TikeReconstructorLibrary:
        core = cls(settings_registry)

        try:
            from .reconstructor import IterativeLeastSquaresReconstructor
            from .reconstructor import RegularizedPIEReconstructor
            from .reconstructor import TikeReconstructor
        except ModuleNotFoundError:
            logger.info('Tike not found.')

            if is_developer_mode_enabled:
                core.reconstructor_list.append(NullReconstructor('rpie'))
                core.reconstructor_list.append(NullReconstructor('lstsq_grad'))
        else:
            tike_version = version('tike')
            logger.info(f'Tike {tike_version}')

            tike_reconstructor = TikeReconstructor(
                core.settings,
                core.multigrid_settings,
                core.position_correction_settings,
                core.probe_correction_settings,
                core.object_correction_settings,
            )
            core.reconstructor_list.append(RegularizedPIEReconstructor(tike_reconstructor))
            core.reconstructor_list.append(IterativeLeastSquaresReconstructor(tike_reconstructor))

        return core

    @property
    def name(self) -> str:
        return 'Tike'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
