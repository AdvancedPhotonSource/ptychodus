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
        self._name.set_value(name)
        self._add_parameter('name', self._name)

    def get_name(self) -> str:
        return self._name.get_value()

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

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
        self.filePath.set_value(filePath)
        self._add_parameter('file_path', self.filePath)
        self.fileType = settings.file_type.copy()
        self.fileType.set_value(fileType)
        self._add_parameter('file_type', self.fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileObjectBuilder:
        return FromFileObjectBuilder(
            self._settings, self.filePath.get_value(), self.fileType.get_value(), self._fileReader
        )

    def build(
        self,
        geometryProvider: ObjectGeometryProvider,
        layerDistanceInMeters: Sequence[float],
    ) -> Object:
        filePath = self.filePath.get_value()
        fileType = self.fileType.get_value()
        logger.debug(f'Reading "{filePath}" as "{fileType}"')

        try:
            objectFromFile = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{filePath}"') from exc

        objectGeometry = geometryProvider.get_object_geometry()
        pixelGeometry = objectFromFile.get_pixel_geometry()
        center = objectFromFile.get_center()

        if pixelGeometry is None:
            pixelGeometry = objectGeometry.get_pixel_geometry()

        if center is None:
            center = objectGeometry.get_center()

        # TODO remap object from pixelGeometryFromFile to pixelGeometryFromProvider
        return Object(
            objectFromFile.get_array(),
            pixelGeometry,
            center,
            objectFromFile.layer_distance_m,
        )
