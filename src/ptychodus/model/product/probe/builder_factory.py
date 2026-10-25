from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
import logging

import numpy

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import (
    OPRWeightPolicy,
    ProbeFileReader,
    ProbeFileWriter,
    ProbeEnsemble,
)
from ptychodus.api.simulate.probe import FresnelZonePlate, KirkpatrickBaezMirrorPair

from ...file_types import FileTypeChooser
from .average_pattern import AveragePatternProbeBuilder
from .builder import FromFileProbeBuilder, ProbeEnsembleBuilder
from .disk import DiskProbeBuilder
from .fzp import FresnelZonePlateProbeBuilder
from .hermite import HermiteProbeBuilder
from .kb_mirror import KBMirrorProbeBuilder
from .rect import RectangularProbeBuilder
from .settings import ProbeSettings
from .super_gaussian import SuperGaussianProbeBuilder
from .zernike import ZernikeProbeBuilder

logger = logging.getLogger(__name__)


class ProbeBuilderFactory(Iterable[str]):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
        kb_mirror_chooser: PluginChooser[KirkpatrickBaezMirrorPair],
        file_reader_chooser: PluginChooser[ProbeFileReader],
        file_writer_chooser: PluginChooser[ProbeFileWriter],
    ) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._fresnel_zone_plate_chooser = fresnel_zone_plate_chooser
        self._kb_mirror_chooser = kb_mirror_chooser
        self.file_readers = FileTypeChooser(file_reader_chooser)
        self.file_writers = FileTypeChooser(file_writer_chooser)
        self._non_diffraction_builders: Mapping[str, Callable[[], ProbeEnsembleBuilder]] = {
            'disk': lambda: DiskProbeBuilder(rng, settings),
            'fresnel_zone_plate': self._create_fresnel_zone_plate_builder,
            'hermite': lambda: HermiteProbeBuilder(rng, settings),
            'kb_mirror': self._create_kb_mirror_builder,
            'rectangular': lambda: RectangularProbeBuilder(rng, settings),
            'super_gaussian': lambda: SuperGaussianProbeBuilder(rng, settings),
            'zernike': lambda: ZernikeProbeBuilder(rng, settings),
        }
        # Builders that cannot build without diffraction data, which they read from the
        # geometry provider when they build.
        self._diffraction_builders: Mapping[str, Callable[[], ProbeEnsembleBuilder]] = {
            'average_pattern': lambda: AveragePatternProbeBuilder(rng, settings),
        }

    def __iter__(self) -> Iterator[str]:
        yield from self._non_diffraction_builders
        yield from self._diffraction_builders

    def create(self, name: str, *, has_diffraction_data: bool = False) -> ProbeEnsembleBuilder:
        """Create the builder *name*, refusing one that needs data on a product without any."""
        diffraction_factory = self._diffraction_builders.get(name)

        if diffraction_factory is not None:
            if not has_diffraction_data:
                raise RuntimeError(
                    f'Probe builder "{name}" requires an associated diffraction dataset.'
                )

            return diffraction_factory()

        try:
            factory = self._non_diffraction_builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown probe builder "{name}"!') from exc

        return factory()

    def create_default(self) -> ProbeEnsembleBuilder:
        return next(iter(self._non_diffraction_builders.values()))()

    def create_from_settings(self, *, has_diffraction_data: bool = False) -> ProbeEnsembleBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_probe_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired, has_diffraction_data=has_diffraction_data)

    def _create_fresnel_zone_plate_builder(self) -> ProbeEnsembleBuilder:
        return FresnelZonePlateProbeBuilder(
            self._rng, self._settings, self._fresnel_zone_plate_chooser
        )

    def _create_kb_mirror_builder(self) -> ProbeEnsembleBuilder:
        return KBMirrorProbeBuilder(self._rng, self._settings, self._kb_mirror_chooser)

    def read_probe_from_file(self, file_path: Path, file_type: str) -> ProbeEnsemble:
        """Read a probe without binding it to a repository item.

        This is the probe the builder from `create_probe_from_file` would ingest, read
        once so a caller can inspect what the file holds -- an OPR basis, in particular --
        before deciding how to ingest it. Read errors surface here rather than being
        swallowed by a rebuild.
        """
        plugin = self.file_readers.select(file_type)
        logger.debug(f'Reading "{file_path}" as "{plugin.simple_name}"')
        return plugin.strategy.read(file_path)

    def create_probe_from_file(
        self,
        file_path: Path,
        file_type: str,
        *,
        opr_weight_policy: OPRWeightPolicy | None = None,
    ) -> ProbeEnsembleBuilder:
        plugin = self.file_readers.select(file_type)

        builder = FromFileProbeBuilder(
            self._rng, self._settings, plugin.strategy, opr_weight_policy=opr_weight_policy
        )
        builder.file_path.set_value(file_path)
        builder.file_type.set_value(plugin.simple_name)
        return builder

    def save_probe(self, file_path: Path, file_type: str, probe: ProbeEnsemble) -> None:
        plugin = self.file_writers.select(file_type)
        logger.debug(f'Writing "{file_path}" as "{plugin.simple_name}"')
        plugin.strategy.write(file_path, probe)
