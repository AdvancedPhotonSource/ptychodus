from collections.abc import Iterator
import logging

from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from ..patterns import Detector
from .device import PtyChiDeviceRepository
from .enums import PtyChiEnumerators
from .settings import (
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiReconstructorSettings,
)

logger = logging.getLogger(__name__)


class PtyChiReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self, settingsRegistry: SettingsRegistry, detector: Detector, isDeveloperModeEnabled: bool
    ) -> None:
        super().__init__()
        self.reconstructorSettings = PtyChiReconstructorSettings(settingsRegistry)
        self.objectSettings = PtyChiObjectSettings(settingsRegistry)
        self.probeSettings = PtyChiProbeSettings(settingsRegistry)
        self.probePositionSettings = PtyChiProbePositionSettings(settingsRegistry)
        self.oprSettings = PtyChiOPRSettings(settingsRegistry)
        self.enumerators = PtyChiEnumerators()
        self.deviceRepository = PtyChiDeviceRepository(
            isDeveloperModeEnabled=isDeveloperModeEnabled
        )
        self.reconstructor_list: list[Reconstructor] = list()

        try:
            from .autodiff import AutodiffReconstructor
            from .lsqml import LSQMLReconstructor
            from .pie import PIEReconstructor
        except ModuleNotFoundError:
            logger.info('pty-chi not found.')

            if isDeveloperModeEnabled:
                for reconstructor in ('PIE', 'ePIE', 'rPIE', 'LSQML', 'Autodiff'):
                    self.reconstructor_list.append(NullReconstructor(reconstructor))
        else:
            self.reconstructor_list.append(
                PIEReconstructor(
                    self.reconstructorSettings,
                    self.objectSettings,
                    self.probeSettings,
                    self.probePositionSettings,
                    self.oprSettings,
                    detector,
                )
            )
            self.reconstructor_list.append(
                LSQMLReconstructor(
                    self.reconstructorSettings,
                    self.objectSettings,
                    self.probeSettings,
                    self.probePositionSettings,
                    self.oprSettings,
                    detector,
                )
            )
            self.reconstructor_list.append(
                AutodiffReconstructor(
                    self.reconstructorSettings,
                    self.objectSettings,
                    self.probeSettings,
                    self.probePositionSettings,
                    self.oprSettings,
                    detector,
                )
            )

    @property
    def name(self) -> str:
        return 'pty-chi'

    @property
    def logger_name(self) -> str:
        return 'ptychi'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
