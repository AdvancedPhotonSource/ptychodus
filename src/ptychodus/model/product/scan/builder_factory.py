from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.scan import PositionSequence, PositionFileReader, PositionFileWriter

from .builder import FromFileScanBuilder, ScanBuilder
from .cartesian import CartesianScanBuilder, CartesianScanVariant
from .concentric import ConcentricScanBuilder
from .lissajous import LissajousScanBuilder
from .settings import ScanSettings
from .spiral import SpiralScanBuilder

logger = logging.getLogger(__name__)


class ScanBuilderFactory(Iterable[str]):
    def __init__(
        self,
        settings: ScanSettings,
        file_reader_chooser: PluginChooser[PositionFileReader],
        file_writer_chooser: PluginChooser[PositionFileWriter],
    ) -> None:
        self._settings = settings
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._builders: dict[str, Callable[[], ScanBuilder]] = {
            variant.name.lower(): lambda variant=variant: CartesianScanBuilder(variant, settings)  # type: ignore
            for variant in CartesianScanVariant
        }
        self._builders.update(
            {
                'concentric': lambda: ConcentricScanBuilder(settings),
                'spiral': lambda: SpiralScanBuilder(settings),
                'lissajous': lambda: LissajousScanBuilder(settings),
            }
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str) -> ScanBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown scan builder "{name}"!') from exc

        return factory()

    def create_default(self) -> ScanBuilder:
        return next(iter(self._builders.values()))()

    def create_from_settings(self) -> ScanBuilder:
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

    def create_scan_from_file(self, file_path: Path, file_type: str) -> ScanBuilder:
        self._file_reader_chooser.set_current_plugin(file_type)
        file_type = self._file_reader_chooser.get_current_plugin().simple_name
        file_reader = self._file_reader_chooser.get_current_plugin().strategy
        return FromFileScanBuilder(self._settings, file_path, file_type, file_reader)

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def save_scan(self, file_path: Path, file_type: str, scan: PositionSequence) -> None:
        self._file_writer_chooser.set_current_plugin(file_type)
        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        file_writer = self._file_writer_chooser.get_current_plugin().strategy
        file_writer.write(file_path, scan)
