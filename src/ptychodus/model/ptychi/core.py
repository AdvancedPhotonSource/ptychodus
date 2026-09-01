from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
import logging
from typing import TYPE_CHECKING, Any

from ptychodus.api.product import Product
from ptychodus.api.reconstruct import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from ._names import DISPLAY_NAMES
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

    from .algorithms import PtyChiAlgorithm

logger = logging.getLogger(__name__)


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
        self._algorithms: dict[str, PtyChiAlgorithm[Any]] | None = None

        # One try/except/else: import the minimal pty-chi surface needed to
        # construct the wrappers (ptychi.api pulls torch for type annotations
        # only -- no CUDA context), then either build the real reconstructors
        # or fall back to NullReconstructors.
        try:
            from .algorithms import PtyChiCommon, build_algorithms, wrap_as_subprocess_reconstructor
        except ImportError:
            logger.info('pty-chi not found.')
            if is_developer_mode_enabled:
                for display_name in DISPLAY_NAMES:
                    self.reconstructor_list.append(NullReconstructor(display_name))
        else:
            try:
                ptychi_version = version('ptychi')
            except PackageNotFoundError:
                ptychi_version = 'unknown'
            logger.info(f'Pty-Chi {ptychi_version}')

            common = PtyChiCommon(
                self.settings,
                self.object_settings,
                self.probe_settings,
                self.probe_position_settings,
                self.opr_settings,
            )
            self._algorithms = build_algorithms(
                common,
                dm_settings=self.dm_settings,
                raar_settings=self.raar_settings,
                pie_settings=self.pie_settings,
                lsqml_settings=self.lsqml_settings,
                autodiff_settings=self.autodiff_settings,
                bh_settings=self.bh_settings,
            )
            self.reconstructor_list.extend(
                wrap_as_subprocess_reconstructor(algorithm, common)
                for algorithm in self._algorithms.values()
            )

    def build_task_options(self, algorithm_name: str, product: Product) -> PtychographyTaskOptions:
        """Build a pty-chi ``PtychographyTaskOptions`` for the named algorithm.

        Public entry point for out-of-process launchers (see
        ``scripts/ptychodus_reconstruct_parent_demo.py``). Case-insensitive on
        the display name, so 'rPIE' and 'rpie' both resolve.
        """
        if self._algorithms is None:
            raise RuntimeError('pty-chi is not available; cannot build task options.')
        try:
            algorithm = self._algorithms[algorithm_name.casefold()]
        except KeyError:
            raise KeyError(f'Unknown pty-chi algorithm "{algorithm_name}"!') from None
        return algorithm.build_task_options(product)

    @property
    def name(self) -> str:
        return 'pty-chi'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
