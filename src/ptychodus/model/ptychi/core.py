from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
import logging
from typing import TYPE_CHECKING

from ptychodus.api.product import Product
from ptychodus.api.reconstruct import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from .device import PtyChiDeviceRepository
from .enums import PtyChiEnumerators
from .settings import (
    PtyChiAutodiffSettings,
    PtyChiBHSettings,
    PtyChiDMSettings,
    PtyChiLSQMLSettings,
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiRAARSettings,
    PtyChiSettings,
)

if TYPE_CHECKING:
    from ptychi.api.options.task import PtychographyTaskOptions

    from .algorithms import PtyChiCommon

logger = logging.getLogger(__name__)


# Kept in sync with the display names in algorithms.py::ALGORITHMS. Duplicated
# here because algorithms.py imports ptychi.api, which is what the try/except
# below is guarding against.
_DEVELOPER_MODE_DISPLAY_NAMES = ('DM', 'RAAR', 'PIE', 'ePIE', 'rPIE', 'LSQML', 'Autodiff', 'BH')


class PtyChiReconstructorLibrary(ReconstructorLibrary):
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__('ptychi')
        self.autodiff_settings = PtyChiAutodiffSettings(settings_registry)
        self.bh_settings = PtyChiBHSettings(settings_registry)
        self.dm_settings = PtyChiDMSettings(settings_registry)
        self.lsqml_settings = PtyChiLSQMLSettings(settings_registry)
        self.object_settings = PtyChiObjectSettings(settings_registry)
        self.opr_settings = PtyChiOPRSettings(settings_registry)
        self.pie_settings = PtyChiPIESettings(settings_registry)
        self.probe_position_settings = PtyChiProbePositionSettings(settings_registry)
        self.probe_settings = PtyChiProbeSettings(settings_registry)
        self.raar_settings = PtyChiRAARSettings(settings_registry)
        self.settings = PtyChiSettings(settings_registry)

        self.enumerators = PtyChiEnumerators()
        self.device_repository = PtyChiDeviceRepository(
            is_developer_mode_enabled=is_developer_mode_enabled
        )
        self.reconstructor_list: list[Reconstructor] = list()
        self._common: 'PtyChiCommon | None' = None

        # One try/except/else: import the minimal pty-chi surface needed to
        # construct the wrappers (ptychi.api pulls torch for type annotations
        # only -- no CUDA context), then either build the real reconstructors
        # or fall back to NullReconstructors.
        try:
            from .algorithms import (
                PtyChiCommon,
                create_autodiff_reconstructor,
                create_bh_reconstructor,
                create_dm_reconstructor,
                create_epie_reconstructor,
                create_lsqml_reconstructor,
                create_pie_reconstructor,
                create_raar_reconstructor,
                create_rpie_reconstructor,
            )
        except ImportError:
            logger.info('pty-chi not found.')
            if is_developer_mode_enabled:
                for display_name in _DEVELOPER_MODE_DISPLAY_NAMES:
                    self.reconstructor_list.append(NullReconstructor(display_name))
        else:
            try:
                ptychi_version = version('ptychi')
            except PackageNotFoundError:
                ptychi_version = 'unknown'
            logger.info(f'Pty-Chi {ptychi_version}')

            self._common = PtyChiCommon(
                self.settings,
                self.object_settings,
                self.probe_settings,
                self.probe_position_settings,
                self.opr_settings,
            )
            self.reconstructor_list.extend(
                [
                    create_dm_reconstructor(self._common, self.dm_settings),
                    create_raar_reconstructor(self._common, self.raar_settings),
                    create_pie_reconstructor(self._common, self.pie_settings),
                    create_epie_reconstructor(self._common, self.pie_settings),
                    create_rpie_reconstructor(self._common, self.pie_settings),
                    create_lsqml_reconstructor(self._common, self.lsqml_settings),
                    create_autodiff_reconstructor(self._common, self.autodiff_settings),
                    create_bh_reconstructor(self._common, self.bh_settings),
                ]
            )

    def build_task_options(
        self, algorithm_name: str, product: Product
    ) -> 'PtychographyTaskOptions':
        """Build a pty-chi ``PtychographyTaskOptions`` for the named algorithm.

        Public entry point for out-of-process launchers (see
        ``scripts/ptychodus_reconstruct_parent_demo.py``). Case-insensitive on
        the display name, so 'rPIE' and 'rpie' both resolve.
        """
        if self._common is None:
            raise RuntimeError('pty-chi is not available; cannot build task options.')

        from .algorithms import build_task_options_for_algorithm

        return build_task_options_for_algorithm(
            self._common,
            algorithm_name,
            product,
            dm_settings=self.dm_settings,
            raar_settings=self.raar_settings,
            pie_settings=self.pie_settings,
            lsqml_settings=self.lsqml_settings,
            autodiff_settings=self.autodiff_settings,
            bh_settings=self.bh_settings,
        )

    @property
    def name(self) -> str:
        return 'pty-chi'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
