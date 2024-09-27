from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.object import Object, ObjectFileReader, ObjectGeometryProvider
from ptychodus.api.parametric import ParameterGroup, PathParameter, StringParameter

logger = logging.getLogger(__name__)


class ObjectBuilder(ParameterGroup):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = StringParameter(self, "name", name)

    def getName(self) -> str:
        return self._name.getValue()

    @abstractmethod
    def copy(self) -> ObjectBuilder:
        pass

    @abstractmethod
    def build(
        self,
        geometryProvider: ObjectGeometryProvider,
        layerDistanceInMeters: Sequence[float],
    ) -> Object:
        pass


class FromMemoryObjectBuilder(ObjectBuilder):
    def __init__(self, object_: Object) -> None:
        super().__init__("from_memory")
        self._object = object_.copy()

    def copy(self) -> FromMemoryObjectBuilder:
        return FromMemoryObjectBuilder(self._object)

    def build(
        self,
        geometryProvider: ObjectGeometryProvider,
        layerDistanceInMeters: Sequence[float],
    ) -> Object:
        return self._object


class FromFileObjectBuilder(ObjectBuilder):
    def __init__(self, filePath: Path, fileType: str, fileReader: ObjectFileReader) -> None:
        super().__init__("from_file")
        self.filePath = PathParameter(self, "file_path", filePath)
        self.fileType = StringParameter(self, "file_type", fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileObjectBuilder:
        return FromFileObjectBuilder(
            self.filePath.getValue(), self.fileType.getValue(), self._fileReader
        )

    def build(
        self,
        geometryProvider: ObjectGeometryProvider,
        layerDistanceInMeters: Sequence[float],
    ) -> Object:
        filePath = self.filePath.getValue()
        fileType = self.fileType.getValue()
        logger.debug(f'Reading "{filePath}" as "{fileType}"')

        try:
            object_ = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{filePath}"') from exc

        return object_
