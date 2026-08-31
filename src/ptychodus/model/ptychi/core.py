from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import logging

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

logger = logging.getLogger(__name__)


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

        if not _ptychi_available():
            logger.info('pty-chi not found.')

            if is_developer_mode_enabled:
                for reconstructor in (
                    'DM',
                    'RAAR',
                    'PIE',
                    'ePIE',
                    'rPIE',
                    'LSQML',
                    'Autodiff',
                    'BH',
                ):
                    self.reconstructor_list.append(NullReconstructor(reconstructor))
            return

        try:
            ptychi_version = version('ptychi')
        except PackageNotFoundError:
            ptychi_version = 'unknown'
        logger.info(f'Pty-Chi {ptychi_version}')

        # Parent-side factory. Imports ptychi.api transitively (via .helper and
        # per-algorithm modules) — that pulls torch but does not acquire a GPU
        # context; see the invariant note in _subprocess_protocol.py.
        from .reconstructor import PtyChiSettingsBundle, build_reconstructor_list

        bundle = PtyChiSettingsBundle(
            dm=self.dm_settings,
            raar=self.raar_settings,
            pie=self.pie_settings,
            lsqml=self.lsqml_settings,
            autodiff=self.autodiff_settings,
            bh=self.bh_settings,
        )
        self.reconstructor_list.extend(
            build_reconstructor_list(
                self.settings,
                self.object_settings,
                self.probe_settings,
                self.probe_position_settings,
                self.opr_settings,
                bundle,
            )
        )

    @property
    def name(self) -> str:
        return 'pty-chi'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
