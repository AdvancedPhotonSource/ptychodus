from collections.abc import Iterator
from importlib.metadata import version
import logging

from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import PatternSizer
from .device import PtyChiDeviceRepository
from .enums import PtyChiEnumerators
from .settings import (
    PtyChiAutodiffSettings,
    PtyChiDMSettings,
    PtyChiLSQMLSettings,
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiSettings,
)

logger = logging.getLogger(__name__)


class PtyChiReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        pattern_sizer: PatternSizer,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__('ptychi')
        self.autodiff_settings = PtyChiAutodiffSettings(settings_registry)
        self.dm_settings = PtyChiDMSettings(settings_registry)
        self.lsqml_settings = PtyChiLSQMLSettings(settings_registry)
        self.object_settings = PtyChiObjectSettings(settings_registry)
        self.opr_settings = PtyChiOPRSettings(settings_registry)
        self.pie_settings = PtyChiPIESettings(settings_registry)
        self.probe_position_settings = PtyChiProbePositionSettings(settings_registry)
        self.probe_settings = PtyChiProbeSettings(settings_registry)
        self.settings = PtyChiSettings(settings_registry)

        self.enumerators = PtyChiEnumerators()
        self.device_repository = PtyChiDeviceRepository(
            is_developer_mode_enabled=is_developer_mode_enabled
        )
        self.reconstructor_list: list[Reconstructor] = list()

        try:
            from .autodiff import AutodiffReconstructor
            from .dm import DMReconstructor
            from .epie import EPIEReconstructor
            from .helper import PtyChiOptionsHelper
            from .lsqml import LSQMLReconstructor
            from .pie import PIEReconstructor
            from .rpie import RPIEReconstructor
        except ModuleNotFoundError:
            logger.info('pty-chi not found.')

            if is_developer_mode_enabled:
                for reconstructor in ('DM', 'PIE', 'ePIE', 'rPIE', 'LSQML', 'Autodiff'):
                    self.reconstructor_list.append(NullReconstructor(reconstructor))
        else:
            ptychi_version = version('ptychi')
            logger.info(f'Pty-Chi {ptychi_version}')

            options_helper = PtyChiOptionsHelper(
                self.settings,
                self.object_settings,
                self.probe_settings,
                self.probe_position_settings,
                self.opr_settings,
                pattern_sizer,
            )
            self.reconstructor_list.append(DMReconstructor(options_helper, self.dm_settings))
            self.reconstructor_list.append(PIEReconstructor(options_helper, self.pie_settings))
            self.reconstructor_list.append(EPIEReconstructor(options_helper, self.pie_settings))
            self.reconstructor_list.append(RPIEReconstructor(options_helper, self.pie_settings))
            self.reconstructor_list.append(LSQMLReconstructor(options_helper, self.lsqml_settings))
            self.reconstructor_list.append(
                AutodiffReconstructor(options_helper, self.autodiff_settings)
            )

    @property
    def name(self) -> str:
        return 'pty-chi'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
