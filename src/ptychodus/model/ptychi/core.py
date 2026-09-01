from collections.abc import Iterator, Mapping
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import logging
from typing import TYPE_CHECKING, Any

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
    from ptychi.api import Reconstructors
    from ptychi.api.options.task import PtychographyTaskOptions

    from .algorithms import PtyChiCommon

logger = logging.getLogger(__name__)


# Kept in sync with the display names in algorithms.py::ALGORITHMS. Duplicated
# here because algorithms.py imports ptychi.api, which is exactly what
# _ptychi_available() is guarding against.
_DEVELOPER_MODE_DISPLAY_NAMES = ('DM', 'RAAR', 'PIE', 'ePIE', 'rPIE', 'LSQML', 'Autodiff', 'BH')


def _ptychi_available() -> bool:
    """Return True iff ``ptychi`` is importable, without importing it."""
    return find_spec('ptychi') is not None


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
        self._settings_by_reconstructor: 'Mapping[Reconstructors, Any]' = {}

        if not _ptychi_available():
            logger.info('pty-chi not found.')

            if is_developer_mode_enabled:
                for display_name in _DEVELOPER_MODE_DISPLAY_NAMES:
                    self.reconstructor_list.append(NullReconstructor(display_name))
            return

        try:
            ptychi_version = version('ptychi')
        except PackageNotFoundError:
            ptychi_version = 'unknown'
        logger.info(f'Pty-Chi {ptychi_version}')

        # Deferred import: pulls ptychi.api (via .algorithms) which pulls torch
        # for its type annotations, but acquires no GPU context.
        from .algorithms import PtyChiCommon, build_reconstructor_list

        self._common = PtyChiCommon(
            self.settings,
            self.object_settings,
            self.probe_settings,
            self.probe_position_settings,
            self.opr_settings,
        )
        self._settings_by_reconstructor = self._build_settings_by_reconstructor()
        self.reconstructor_list.extend(
            build_reconstructor_list(self._common, self._settings_by_reconstructor)
        )

    def _build_settings_by_reconstructor(self) -> 'Mapping[Reconstructors, Any]':
        # Local import to keep ptychi off the module-level path.
        from ptychi.api import Reconstructors

        return {
            Reconstructors.DM: self.dm_settings,
            Reconstructors.RAAR: self.raar_settings,
            Reconstructors.PIE: self.pie_settings,
            Reconstructors.EPIE: self.pie_settings,
            Reconstructors.RPIE: self.pie_settings,
            Reconstructors.LSQML: self.lsqml_settings,
            Reconstructors.AD_PTYCHO: self.autodiff_settings,
            Reconstructors.BH: self.bh_settings,
        }

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

        from .algorithms import ALGORITHMS

        for reconstructor, algo_cls in ALGORITHMS.items():
            if algo_cls.spec.display_name.casefold() == algorithm_name.casefold():
                algorithm = algo_cls(self._common, self._settings_by_reconstructor[reconstructor])
                return algorithm.build_task_options(product)

        raise KeyError(f'Unknown pty-chi algorithm "{algorithm_name}"!')

    @property
    def name(self) -> str:
        return 'pty-chi'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
