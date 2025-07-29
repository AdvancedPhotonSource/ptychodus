from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy

from .geometry import PixelGeometry
from .scan import ScanPoint
from .typing import ComplexArrayType


@dataclass(frozen=True)
class ObjectCenter:
    position_x_m: float
    position_y_m: float

    def copy(self) -> ObjectCenter:
        return ObjectCenter(
            position_x_m=float(self.position_x_m),
            position_y_m=float(self.position_y_m),
        )


@dataclass(frozen=True)
class ObjectPoint:
    index: int
    position_x_px: float
    position_y_px: float


@dataclass(frozen=True)
class ObjectGeometry:
    width_px: int
    height_px: int
    pixel_width_m: float
    pixel_height_m: float
    center_x_m: float
    center_y_m: float

    @property
    def width_m(self) -> float:
        return self.width_px * self.pixel_width_m

    @property
    def height_m(self) -> float:
        return self.height_px * self.pixel_height_m

    @property
    def minimum_x_m(self) -> float:
        return self.center_x_m - self.width_m / 2.0

    @property
    def minimum_y_m(self) -> float:
        return self.center_y_m - self.height_m / 2.0

    def get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self.pixel_width_m,
            height_m=self.pixel_height_m,
        )

    def get_center(self) -> ObjectCenter:
        return ObjectCenter(
            position_x_m=self.center_x_m,
            position_y_m=self.center_y_m,
        )

    def map_object_point_to_scan_point(self, point: ObjectPoint) -> ScanPoint:
        rx_px = self.width_px / 2
        ry_px = self.height_px / 2
        dx_m = self.pixel_width_m
        dy_m = self.pixel_height_m

        x_m = self.center_x_m + dx_m * (point.position_x_px - rx_px)
        y_m = self.center_y_m + dy_m * (point.position_y_px - ry_px)

        return ScanPoint(point.index, x_m, y_m)

    def map_scan_point_to_object_point(self, point: ScanPoint) -> ObjectPoint:
        rx_px = self.width_px / 2
        ry_px = self.height_px / 2
        dx_m = self.pixel_width_m
        dy_m = self.pixel_height_m

        x_px = (point.position_x_m - self.center_x_m) / dx_m + rx_px
        y_px = (point.position_y_m - self.center_y_m) / dy_m + ry_px

        return ObjectPoint(point.index, x_px, y_px)

    def contains(self, geometry: ObjectGeometry) -> bool:
        dx = self.center_x_m - geometry.center_x_m
        dy = self.center_y_m - geometry.center_y_m
        dw = self.width_m - geometry.width_m
        dh = self.height_m - geometry.height_m
        return abs(dx) <= dw and abs(dy) <= dh


class ObjectGeometryProvider(ABC):
    @abstractmethod
    def get_object_geometry(self) -> ObjectGeometry:
        pass


class Object:
    def __init__(
        self,
        array: ComplexArrayType | None,
        pixel_geometry: PixelGeometry | None,
        center: ObjectCenter | None,
        layer_spacing_m: Sequence[float] = [],
    ) -> None:
        if array is None:
            self._array: ComplexArrayType = numpy.zeros((1, 0, 0), dtype=complex)
        elif numpy.iscomplexobj(array):
            match array.ndim:
                case 2:
                    self._array = array[numpy.newaxis, ...]
                case 3:
                    self._array = array
                case _:
                    raise ValueError('Object must be 2- or 3-dimensional ndarray.')
        else:
            raise TypeError('Object must be a complex-valued ndarray')

        self._pixel_geometry = pixel_geometry
        self._center = center
        self._layer_spacing_m = layer_spacing_m

        expected_layers = self._array.shape[-3]
        actual_layers = len(layer_spacing_m) + 1

        if actual_layers != expected_layers:
            raise ValueError(f'Expected {expected_layers} layers; got {actual_layers}!')

    def copy(self) -> Object:
        return Object(
            array=self._array.copy(),
            pixel_geometry=None if self._pixel_geometry is None else self._pixel_geometry.copy(),
            center=None if self._center is None else self._center.copy(),
            layer_spacing_m=list(self._layer_spacing_m),
        )

    def get_array(self) -> ComplexArrayType:
        return self._array

    @property
    def dtype(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    @property
    def width_px(self) -> int:
        return self._array.shape[-1]

    @property
    def height_px(self) -> int:
        return self._array.shape[-2]

    @property
    def num_layers(self) -> int:
        return self._array.shape[-3]

    def get_pixel_geometry(self) -> PixelGeometry:
        if self._pixel_geometry is None:
            raise ValueError('Missing object pixel geometry!')

        return self._pixel_geometry

    def get_center(self) -> ObjectCenter:
        if self._center is None:
            raise ValueError('Missing object center!')

        return self._center

    def get_geometry(self) -> ObjectGeometry:
        pixel_geometry = self.get_pixel_geometry()
        center = self.get_center()

        return ObjectGeometry(
            width_px=self.width_px,
            height_px=self.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            center_x_m=center.position_x_m,
            center_y_m=center.position_y_m,
        )

    def get_layer(self, number: int) -> ComplexArrayType:
        return self._array[number, :, :]

    def get_layers_flattened(self) -> ComplexArrayType:
        return numpy.prod(self._array, axis=-3)

    @property
    def layer_spacing_m(self) -> Sequence[float]:
        return self._layer_spacing_m

    def get_total_thickness_m(self) -> float:
        return sum(self._layer_spacing_m)

    def __repr__(self) -> str:
        return f'{self._array.dtype}{self._array.shape}'


class ObjectFileReader(ABC):
    @abstractmethod
    def read(self, file_path: Path) -> Object:
        """reads an object from file"""
        pass


class ObjectFileWriter(ABC):
    @abstractmethod
    def write(self, file_path: Path, object_: Object) -> None:
        """writes an object to file"""
        pass
