from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectFileReader, ObjectGeometryProvider
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.typing import ComplexArrayType

from ...phase_unwrapper import PhaseUnwrapper
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectBuilder(ParameterGroup):
    def __init__(self, settings: ObjectSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

        self.extra_padding_x = settings.extra_padding_x.copy()
        self._add_parameter('extra_padding_x', self.extra_padding_x)
        self.extra_padding_y = settings.extra_padding_y.copy()
        self._add_parameter('extra_padding_y', self.extra_padding_y)

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

    def _create_object(
        self,
        array: ComplexArrayType,
        pixel_geometry: PixelGeometry,
        center: ObjectCenter,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        """Create an object from an existing object with a potentially
        different number of slices.

        If the new object is supposed to be a multislice object with a
        different number of slices than the existing object, the object is
        created as
        `abs(o) ** (1 / nSlices) * exp(i * unwrapPhase(o) / nSlices)`.
        Otherwise, the object is copied as is.
        """
        num_slices = 1 + len(layer_spacing_m)

        if array.ndim < 2:
            raise ValueError('Array must have at least 2 dimensions')
        elif array.ndim == 2:
            array = numpy.expand_dims(array, axis=0)
        elif array.ndim > 3:
            raise ValueError('Array must have at most 3 dimensions')

        if num_slices < array.shape[0]:  # FIXME test
            array = array[0:num_slices]
        elif num_slices > array.shape[0]:  # FIXME test
            amplitude = numpy.absolute(array[0:1]) ** (1.0 / num_slices)
            amplitude = amplitude.repeat(num_slices, axis=0)
            phase = PhaseUnwrapper().unwrap(array[0])[None, ...] / num_slices
            phase = phase.repeat(num_slices, axis=0)
            array = numpy.clip(amplitude, 0.0, 1.0) * numpy.exp(1j * phase)

        pad_width = [
            (0, 0),
            (self.extra_padding_y.get_value(), self.extra_padding_y.get_value()),
            (self.extra_padding_x.get_value(), self.extra_padding_x.get_value()),
        ]
        return Object(
            array=numpy.pad(array, pad_width),
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=pixel_geometry,
            center=center,
        )


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
