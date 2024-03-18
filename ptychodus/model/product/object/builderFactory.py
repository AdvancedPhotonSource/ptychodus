from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
import logging

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter, ObjectGeometryProvider
from ptychodus.api.plugins import PluginChooser

from .builder import FromFileObjectBuilder, ObjectBuilder
from .random import RandomObjectBuilder

logger = logging.getLogger(__name__)


class ObjectBuilderFactory(Iterable[str]):

    def __init__(self, rng: numpy.random.Generator,
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        self._rng = rng
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._builders: Mapping[str, Callable[[ObjectGeometryProvider], ObjectBuilder]] = {
            'random': self._createRandomBuilder,
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str, geometryProvider: ObjectGeometryProvider) -> ObjectBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown object builder \"{name}\"!') from exc

        return factory(geometryProvider)

    def _createRandomBuilder(self, geometryProvider: ObjectGeometryProvider) -> ObjectBuilder:
        return RandomObjectBuilder(self._rng, geometryProvider)

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
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        fileWriter = self._fileWriterChooser.currentPlugin.strategy
        fileWriter.write(filePath, object_)
