from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
import logging

import numpy

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePositionFileWriter,
)

from .builder import (
    FromFileProbePositionsBuilder,
    FromMemoryProbePositionsBuilder,
    ProbePositionsBuilder,
)
from .cartesian import CartesianProbePositionsBuilder, CartesianProbePositionsVariant
from .concentric import ConcentricProbePositionsBuilder
from .lissajous import LissajousProbePositionsBuilder
from .settings import ProbePositionsSettings
from .spiral import SpiralProbePositionsBuilder

logger = logging.getLogger(__name__)


class ProbePositionsBuilderFactory(Iterable[str]):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        file_reader_chooser: PluginChooser[ProbePositionFileReader],
        file_writer_chooser: PluginChooser[ProbePositionFileWriter],
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._builders: dict[str, Callable[[], ProbePositionsBuilder]] = {
            variant.name.lower(): lambda variant=variant: self._create_cartesian_builder(variant)
            for variant in CartesianProbePositionsVariant
        }
        self._builders.update(
            {
                'concentric': lambda: ConcentricProbePositionsBuilder(rng, settings),
                'spiral': lambda: SpiralProbePositionsBuilder(rng, settings),
                'lissajous': lambda: LissajousProbePositionsBuilder(rng, settings),
            }
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str) -> ProbePositionsBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown scan builder "{name}"!') from exc

        return factory()

    def create_default(self) -> ProbePositionsBuilder:
        return next(iter(self._builders.values()))()

    def create_from_settings(self) -> ProbePositionsBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_scan_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired)

    def _create_cartesian_builder(
        self, variant: CartesianProbePositionsVariant
    ) -> CartesianProbePositionsBuilder:
        return CartesianProbePositionsBuilder(variant, self._rng, self._settings)

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def create_scan_from_memory(self, position_seq: ProbePositionSequence) -> ProbePositionsBuilder:
        return FromMemoryProbePositionsBuilder(self._rng, self._settings, position_seq)

    def create_scan_from_file(self, file_path: Path, file_type: str) -> ProbePositionsBuilder:
        self._file_reader_chooser.set_current_plugin(file_type)
        file_reader = self._file_reader_chooser.get_current_plugin().strategy

        builder = FromFileProbePositionsBuilder(self._rng, self._settings, file_reader)
        builder.file_path.set_value(file_path)
        builder.file_type.set_value(self._file_reader_chooser.get_current_plugin().simple_name)
        return builder

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def save_scan(
        self, file_path: Path, file_type: str, position_seq: ProbePositionSequence
    ) -> None:
        self._file_writer_chooser.set_current_plugin(file_type)
        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        file_writer = self._file_writer_chooser.get_current_plugin().strategy
        file_writer.write(file_path, position_seq)
