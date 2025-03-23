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
        fileReaderChooser: PluginChooser[ObjectFileReader],
        fileWriterChooser: PluginChooser[ObjectFileWriter],
    ) -> None:
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
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

    def createDefault(self) -> ObjectBuilder:
        return next(iter(self._builders.values()))()

    def createFromSettings(self) -> ObjectBuilder:
        name = self._settings.builder.get_value()
        nameRepaired = name.casefold()

        if nameRepaired == 'from_file':
            return self.createObjectFromFile(
                self._settings.filePath.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(nameRepaired)

    def getOpenFileFilterList(self) -> Iterator[str]:
        for plugin in self._fileReaderChooser:
            yield plugin.display_name

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.get_current_plugin().display_name

    def createObjectFromFile(self, filePath: Path, fileFilter: str) -> ObjectBuilder:
        self._fileReaderChooser.set_current_plugin(fileFilter)
        fileType = self._fileReaderChooser.get_current_plugin().simple_name
        fileReader = self._fileReaderChooser.get_current_plugin().strategy
        return FromFileObjectBuilder(self._settings, filePath, fileType, fileReader)

    def getSaveFileFilterList(self) -> Iterator[str]:
        for plugin in self._fileWriterChooser:
            yield plugin.display_name

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.get_current_plugin().display_name

    def saveObject(self, filePath: Path, fileFilter: str, object_: Object) -> None:
        self._fileWriterChooser.set_current_plugin(fileFilter)
        fileType = self._fileWriterChooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        fileWriter = self._fileWriterChooser.get_current_plugin().strategy
        fileWriter.write(filePath, object_)
