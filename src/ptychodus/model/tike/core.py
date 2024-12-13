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
    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        super().__init__()
        self.settings = TikeSettings(settingsRegistry)
        self.multigridSettings = TikeMultigridSettings(settingsRegistry)
        self.positionCorrectionSettings = TikePositionCorrectionSettings(settingsRegistry)
        self.probeCorrectionSettings = TikeProbeCorrectionSettings(settingsRegistry)
        self.objectCorrectionSettings = TikeObjectCorrectionSettings(settingsRegistry)

        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(
        cls, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool
    ) -> TikeReconstructorLibrary:
        core = cls(settingsRegistry)

        try:
            from .reconstructor import IterativeLeastSquaresReconstructor
            from .reconstructor import RegularizedPIEReconstructor
            from .reconstructor import TikeReconstructor
        except ModuleNotFoundError:
            logger.info('Tike not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('rpie'))
                core.reconstructorList.append(NullReconstructor('lstsq_grad'))
        else:
            tikeVersion = version('tike')
            logger.info(f'Tike {tikeVersion}')

            tikeReconstructor = TikeReconstructor(
                core.settings,
                core.multigridSettings,
                core.positionCorrectionSettings,
                core.probeCorrectionSettings,
                core.objectCorrectionSettings,
            )
            core.reconstructorList.append(RegularizedPIEReconstructor(tikeReconstructor))
            core.reconstructorList.append(IterativeLeastSquaresReconstructor(tikeReconstructor))

        return core

    @property
    def name(self) -> str:
        return 'Tike'

    @property
    def logger_name(self) -> str:
        return 'tike'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructorList)
