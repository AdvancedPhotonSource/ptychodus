from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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
            "random": lambda: RandomObjectBuilder(rng, settings),
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
        name = self._settings.builder.getValue()
        nameRepaired = name.casefold()

        if nameRepaired == "from_file":
            return self.createObjectFromFile(
                self._settings.filePath.getValue(),
                self._settings.fileType.getValue(),
            )

        return self.create(nameRepaired)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def createObjectFromFile(self, filePath: Path, fileFilter: str) -> ObjectBuilder:
        self._fileReaderChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileReaderChooser.currentPlugin.simpleName
        fileReader = self._fileReaderChooser.currentPlugin.strategy
        return FromFileObjectBuilder(filePath, fileType, fileReader)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveObject(self, filePath: Path, fileFilter: str, object_: Object) -> None:
        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        fileWriter = self._fileWriterChooser.currentPlugin.strategy
        fileWriter.write(filePath, object_)
