from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final
import logging

import numpy

from ptychodus.api.parameters import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import (
    OPRWeightPolicy,
    ProbeFileReader,
    ProbeFileWriter,
    ProbeEnsemble,
)
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    KirkpatrickBaezMirrorPreset,
    ProbeGenerator,
)

from ...declared import describe
from ...file_types import FileTypeChooser
from .builder import FromFileProbeBuilder, ProbeEnsembleBuilder
from .plugin import PluginProbeBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeBuilderFactory(Iterable[str]):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        generator_chooser: PluginChooser[ProbeGenerator],
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
        kb_mirror_chooser: PluginChooser[KirkpatrickBaezMirrorPreset],
        file_reader_chooser: PluginChooser[ProbeFileReader],
        file_writer_chooser: PluginChooser[ProbeFileWriter],
    ) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._generators = generator_chooser
        self._fresnel_zone_plate_chooser = fresnel_zone_plate_chooser
        self._kb_mirror_chooser = kb_mirror_chooser
        self.file_readers = FileTypeChooser(file_reader_chooser)
        self.file_writers = FileTypeChooser(file_writer_chooser)

        # Every registered generator, not only the one selected: the settings file holds
        # one section for the probe and it must hold the same keys however the
        # application was driven, so a model never chosen still contributes the keys it
        # would be configured by.
        self._parameters: dict[str, Mapping[str, Parameter[Any]]] = {
            plugin.simple_name: settings.declare(
                describe(plugin.strategy.get_default_options(), expand_num=True)
            )
            for plugin in generator_chooser
        }

    def __iter__(self) -> Iterator[str]:
        for plugin in self._generators:
            yield plugin.simple_name

    def create(self, name: str, *, has_diffraction_data: bool = False) -> ProbeEnsembleBuilder:
        """Create the builder *name*, refusing one that needs data on a product without any."""
        plugin = self._generators.find_plugin(name)

        if plugin is None:
            raise KeyError(f'Unknown probe builder "{name}"!')

        if plugin.strategy.reads_diffraction_data and not has_diffraction_data:
            raise RuntimeError(
                f'Probe builder "{name}" requires an associated diffraction dataset.'
            )

        return PluginProbeBuilder(
            self._rng,
            self._settings,
            plugin.simple_name,
            plugin.strategy,
            self._parameters[plugin.simple_name],
            self._fresnel_zone_plate_chooser,
            self._kb_mirror_chooser,
        )

    def create_default(self) -> ProbeEnsembleBuilder:
        # Named rather than taken as the first the chooser holds: the chooser sorts by
        # name, so that would silently make it whichever model sorts first.
        default_generator: Final = 'disk'
        return self.create(default_generator)

    def create_from_settings(self, *, has_diffraction_data: bool = False) -> ProbeEnsembleBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_probe_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired, has_diffraction_data=has_diffraction_data)

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
