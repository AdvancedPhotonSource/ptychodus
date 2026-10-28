from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final
import logging

import numpy

from ptychodus.api.parameters import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePositionFileWriter,
)
from ptychodus.api.simulate.probe_positions import ProbePositionsGenerator

from ...declared import describe
from ...file_types import FileTypeChooser
from .builder import (
    FromFileProbePositionsBuilder,
    FromMemoryProbePositionsBuilder,
    ProbePositionsBuilder,
)
from .plugin import PluginProbePositionsBuilder
from .settings import ProbePositionsSettings

logger = logging.getLogger(__name__)

# The pattern a fresh install starts on, and the fallback when a settings file
# names one no registered generator answers to. Named rather than taken as the
# first the chooser holds: the chooser sorts by name, so that would silently
# make it whichever pattern sorts first.
_DEFAULT_GENERATOR: Final = 'rectangular_raster'


class ProbePositionsBuilderFactory(Iterable[str]):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        generator_chooser: PluginChooser[ProbePositionsGenerator],
        file_reader_chooser: PluginChooser[ProbePositionFileReader],
        file_writer_chooser: PluginChooser[ProbePositionFileWriter],
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._generators = generator_chooser
        self.file_readers = FileTypeChooser(file_reader_chooser)
        self.file_writers = FileTypeChooser(file_writer_chooser)

        # Every registered generator, not only the one selected: the settings
        # file holds one section for scan positions and it must hold the same
        # keys however the application was driven, so a pattern never chosen
        # still contributes the keys it would be configured by.
        self._parameters: dict[str, Mapping[str, Parameter[Any]]] = {
            plugin.simple_name: settings.declare(
                describe(plugin.strategy.get_default_options(), expand_num=True)
            )
            for plugin in generator_chooser
        }

    def __iter__(self) -> Iterator[str]:
        for plugin in self._generators:
            yield plugin.simple_name

    def create(self, name: str) -> ProbePositionsBuilder:
        plugin = self._generators.find_plugin(name)

        if plugin is None:
            raise KeyError(f'Unknown scan builder "{name}"!')

        return PluginProbePositionsBuilder(
            self._rng,
            self._settings,
            plugin.simple_name,
            plugin.strategy,
            self._parameters[plugin.simple_name],
        )

    def create_default(self) -> ProbePositionsBuilder:
        return self.create(_DEFAULT_GENERATOR)

    def create_from_settings(self) -> ProbePositionsBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_scan_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired)

    def create_scan_from_memory(self, position_seq: ProbePositionSequence) -> ProbePositionsBuilder:
        return FromMemoryProbePositionsBuilder(self._rng, self._settings, position_seq)

    def create_scan_from_file(self, file_path: Path, file_type: str) -> ProbePositionsBuilder:
        plugin = self.file_readers.select(file_type)

        builder = FromFileProbePositionsBuilder(self._rng, self._settings, plugin.strategy)
        builder.file_path.set_value(file_path)
        builder.file_type.set_value(plugin.simple_name)
        return builder

    def save_scan(
        self, file_path: Path, file_type: str, position_seq: ProbePositionSequence
    ) -> None:
        plugin = self.file_writers.select(file_type)
        logger.debug(f'Writing "{file_path}" as "{plugin.simple_name}"')
        plugin.strategy.write(file_path, position_seq)
