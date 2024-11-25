from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.object import Object, ObjectFileReader, ObjectGeometryProvider
from ptychodus.api.parametric import ParameterGroup

from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectBuilder(ParameterGroup):
    def __init__(self, settings: ObjectSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.setValue(name)
        self._addParameter('name', self._name)

    def getName(self) -> str:
        return self._name.getValue()

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

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
    def __init__(self, settings: ObjectSettings, object_: Object) -> None:
        super().__init__(settings, 'from_memory')
        self._settings = settings
        self._object = object_.copy()

    def copy(self) -> FromMemoryObjectBuilder:
        return FromMemoryObjectBuilder(self._settings, self._object)

    def build(
        self,
        geometryProvider: ObjectGeometryProvider,
        layerDistanceInMeters: Sequence[float],
    ) -> Object:
        return self._object


class FromFileObjectBuilder(ObjectBuilder):
    def __init__(
        self, settings: ObjectSettings, filePath: Path, fileType: str, fileReader: ObjectFileReader
    ) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self.filePath = settings.filePath.copy()
        self.filePath.setValue(filePath)
        self._addParameter('file_path', self.filePath)
        self.fileType = settings.fileType.copy()
        self.fileType.setValue(fileType)
        self._addParameter('file_type', self.fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileObjectBuilder:
        return FromFileObjectBuilder(
            self._settings, self.filePath.getValue(), self.fileType.getValue(), self._fileReader
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
            objectFromFile = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{filePath}"') from exc

        pixelGeometryFromFile = objectFromFile.getPixelGeometry()
        pixelGeometryFromProvider = geometryProvider.getObjectGeometry().getPixelGeometry()

        if pixelGeometryFromFile is None:
            return Object(
                objectFromFile.getArray(),
                pixelGeometryFromProvider,
                objectFromFile.layerDistanceInMeters,
                centerXInMeters=objectFromFile.centerXInMeters,
                centerYInMeters=objectFromFile.centerYInMeters,
            )

        # TODO remap object from pixelGeometryFromFile to pixelGeometryFromProvider
        return objectFromFile
