from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePositionFileWriter,
)

from .builder import FromFileProbePositionsBuilder, ProbePositionsBuilder
from .cartesian import CartesianProbePositionsBuilder, CartesianProbePositionsVariant
from .concentric import ConcentricProbePositionsBuilder
from .lissajous import LissajousProbePositionsBuilder
from .settings import ProbePositionsSettings
from .spiral import SpiralProbePositionsBuilder

logger = logging.getLogger(__name__)


class ProbePositionsBuilderFactory(Iterable[str]):
    def __init__(
        self,
        settings: ProbePositionsSettings,
        file_reader_chooser: PluginChooser[ProbePositionFileReader],
        file_writer_chooser: PluginChooser[ProbePositionFileWriter],
    ) -> None:
        self._settings = settings
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._builders: dict[str, Callable[[], ProbePositionsBuilder]] = {
            variant.name.lower(): lambda var=variant: CartesianProbePositionsBuilder(var, settings)  # type: ignore
            for variant in CartesianProbePositionsVariant
        }
        self._builders.update(
            {
                'concentric': lambda: ConcentricProbePositionsBuilder(settings),
                'spiral': lambda: SpiralProbePositionsBuilder(settings),
                'lissajous': lambda: LissajousProbePositionsBuilder(settings),
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

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def create_scan_from_file(self, file_path: Path, file_type: str) -> ProbePositionsBuilder:
        self._file_reader_chooser.set_current_plugin(file_type)
        file_type = self._file_reader_chooser.get_current_plugin().simple_name
        file_reader = self._file_reader_chooser.get_current_plugin().strategy
        return FromFileProbePositionsBuilder(self._settings, file_path, file_type, file_reader)

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def save_scan(self, file_path: Path, file_type: str, scan: ProbePositionSequence) -> None:
        self._file_writer_chooser.set_current_plugin(file_type)
        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        file_writer = self._file_writer_chooser.get_current_plugin().strategy
        file_writer.write(file_path, scan)
