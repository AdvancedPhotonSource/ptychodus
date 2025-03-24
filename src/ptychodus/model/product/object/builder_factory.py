from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
import logging

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginChooser

from .builder import FromFileObjectBuilder, ObjectBuilder
from .random import RandomObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectBuilderFactory(Iterable[str]):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ObjectSettings,
        file_reader_chooser: PluginChooser[ObjectFileReader],
        file_writer_chooser: PluginChooser[ObjectFileWriter],
    ) -> None:
        self._settings = settings
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._builders: Mapping[str, Callable[[], ObjectBuilder]] = {
            'random': lambda: RandomObjectBuilder(rng, settings),
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str) -> ObjectBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown object builder "{name}"!') from exc

        return factory()

    def create_default(self) -> ObjectBuilder:
        return next(iter(self._builders.values()))()

    def create_from_settings(self) -> ObjectBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_object_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired)

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def create_object_from_file(self, file_path: Path, file_filter: str) -> ObjectBuilder:
        self._file_reader_chooser.set_current_plugin(file_filter)
        file_type = self._file_reader_chooser.get_current_plugin().simple_name
        file_reader = self._file_reader_chooser.get_current_plugin().strategy
        return FromFileObjectBuilder(self._settings, file_path, file_type, file_reader)

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def save_object(self, file_path: Path, file_filter: str, object_: Object) -> None:
        self._file_writer_chooser.set_current_plugin(file_filter)
        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        file_writer = self._file_writer_chooser.get_current_plugin().strategy
        file_writer.write(file_path, object_)
