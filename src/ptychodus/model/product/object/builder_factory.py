from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Final
import logging

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginCatalog
from ptychodus.api.simulate.object import ObjectGenerator

from ...declared import describe
from ...file_types import FileTypeChooser
from ...parameters import Parameter
from ...plugins import PluginChooser
from .builder import FromFileObjectBuilder, ObjectBuilder
from .plugin import PluginObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectBuilderFactory(Iterable[str]):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ObjectSettings,
        generator_chooser: PluginCatalog[ObjectGenerator],
        file_reader_chooser: PluginChooser[ObjectFileReader],
        file_writer_chooser: PluginChooser[ObjectFileWriter],
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._generators = generator_chooser
        self.file_readers = FileTypeChooser(file_reader_chooser)
        self.file_writers = FileTypeChooser(file_writer_chooser)

        # Every registered generator, not only the one selected: the settings file holds
        # one section for the object and it must hold the same keys however the
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

    def create(self, name: str, *, has_diffraction_data: bool = False) -> ObjectBuilder:
        """Create the builder *name*, refusing one that needs data on a product without any."""
        plugin = self._generators.find_plugin(name)

        if plugin is None:
            raise KeyError(f'Unknown object builder "{name}"!')

        if plugin.strategy.reads_diffraction_data and not has_diffraction_data:
            raise RuntimeError(
                f'Object builder "{name}" requires an associated diffraction dataset.'
            )

        return PluginObjectBuilder(
            self._rng,
            self._settings,
            plugin.simple_name,
            plugin.strategy,
            self._parameters[plugin.simple_name],
        )

    def create_default(self) -> ObjectBuilder:
        # Named rather than taken as the first the chooser holds: the chooser sorts by
        # name, so that would silently make it whichever model sorts first.
        default_generator: Final = 'random'
        return self.create(default_generator)

    def create_from_settings(self, *, has_diffraction_data: bool = False) -> ObjectBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_object_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired, has_diffraction_data=has_diffraction_data)

    def create_object_from_file(self, file_path: Path, file_type: str) -> ObjectBuilder:
        plugin = self.file_readers.select(file_type)

        builder = FromFileObjectBuilder(self._settings, plugin.strategy)
        builder.file_path.set_value(file_path)
        builder.file_type.set_value(plugin.simple_name)
        return builder

    def save_object(self, file_path: Path, file_type: str, object_: Object) -> None:
        plugin = self.file_writers.select(file_type)
        logger.debug(f'Writing "{file_path}" as "{plugin.simple_name}"')
        plugin.strategy.write(file_path, object_)
