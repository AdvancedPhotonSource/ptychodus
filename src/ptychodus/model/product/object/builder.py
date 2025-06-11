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
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
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
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        object_geometry = geometry_provider.get_object_geometry()

        try:
            pixel_geometry = self._object.get_pixel_geometry()
        except ValueError:
            pixel_geometry = object_geometry.get_pixel_geometry()

        try:
            center = self._object.get_center()
        except ValueError:
            center = object_geometry.get_center()

        return Object(
            self._object.get_array(),
            pixel_geometry,
            center,
            self._object.layer_spacing_m,
        )


class FromFileObjectBuilder(ObjectBuilder):
    def __init__(
        self,
        settings: ObjectSettings,
        file_path: Path,
        file_type: str,
        file_reader: ObjectFileReader,
    ) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self.file_path = settings.file_path.copy()
        self.file_path.set_value(file_path)
        self._add_parameter('file_path', self.file_path)
        self.file_type = settings.file_type.copy()
        self.file_type.set_value(file_type)
        self._add_parameter('file_type', self.file_type)
        self._file_reader = file_reader

    def copy(self) -> FromFileObjectBuilder:
        return FromFileObjectBuilder(
            self._settings,
            self.file_path.get_value(),
            self.file_type.get_value(),
            self._file_reader,
        )

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        file_path = self.file_path.get_value()
        file_type = self.file_type.get_value()
        logger.debug(f'Reading "{file_path}" as "{file_type}"')

        try:
            object_from_file = self._file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        object_geometry = geometry_provider.get_object_geometry()

        try:
            pixel_geometry = object_from_file.get_pixel_geometry()
        except ValueError:
            pixel_geometry = object_geometry.get_pixel_geometry()

        try:
            center = object_from_file.get_center()
        except ValueError:
            center = object_geometry.get_center()

        return Object(
            object_from_file.get_array(),
            pixel_geometry,
            center,
            object_from_file.layer_spacing_m,
        )
