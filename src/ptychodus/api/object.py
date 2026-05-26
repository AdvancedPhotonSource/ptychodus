"""Object (transmission function) data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy
from skimage.registration import phase_cross_correlation

from .common import ComplexArrayType, RealArrayType
from .geometry import PixelGeometry, fourier_shift_2d
from .probe_positions import ProbePosition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectCenter:
    """Physical center coordinates of the object array in meters."""

    coordinate_x_m: float
    coordinate_y_m: float

    def copy(self) -> ObjectCenter:
        return ObjectCenter(
            coordinate_x_m=float(self.coordinate_x_m),
            coordinate_y_m=float(self.coordinate_y_m),
        )


@dataclass(frozen=True)
class ObjectPosition:
    """Position expressed in object pixel coordinates."""

    index: int
    coordinate_x_px: float
    coordinate_y_px: float


@dataclass(frozen=True)
class ObjectTransverseCoordinates:
    """2D Cartesian coordinate arrays for the transverse plane of the object, in meters."""

    position_x_m: RealArrayType
    position_y_m: RealArrayType


@dataclass(frozen=True)
class ObjectGeometry:
    """Spatial geometry of the object: size, pixel scale, and center."""

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
            coordinate_x_m=self.center_x_m,
            coordinate_y_m=self.center_y_m,
        )

    def get_transverse_coordinates(self) -> ObjectTransverseCoordinates:
        Y, X = numpy.mgrid[: self.height_px, : self.width_px]  # noqa: N806
        position_x_px = X - (self.width_px - 1) / 2
        position_y_px = Y - (self.height_px - 1) / 2

        position_x_m = position_x_px * self.pixel_width_m
        position_y_m = position_y_px * self.pixel_height_m

        return ObjectTransverseCoordinates(
            position_x_m=position_x_m,
            position_y_m=position_y_m,
        )

    def map_coordinates_object_to_probe(self, position: ObjectPosition) -> ProbePosition:
        rx_px = self.width_px / 2
        ry_px = self.height_px / 2
        dx_m = self.pixel_width_m
        dy_m = self.pixel_height_m

        x_m = self.center_x_m + dx_m * (position.coordinate_x_px - rx_px)
        y_m = self.center_y_m + dy_m * (position.coordinate_y_px - ry_px)

        return ProbePosition(position.index, x_m, y_m)

    def map_coordinates_probe_to_object(self, position: ProbePosition) -> ObjectPosition:
        rx_px = self.width_px / 2
        ry_px = self.height_px / 2
        dx_m = self.pixel_width_m
        dy_m = self.pixel_height_m

        x_px = (position.coordinate_x_m - self.center_x_m) / dx_m + rx_px
        y_px = (position.coordinate_y_m - self.center_y_m) / dy_m + ry_px

        return ObjectPosition(position.index, x_px, y_px)

    def contains(self, geometry: ObjectGeometry) -> bool:
        dx = self.center_x_m - geometry.center_x_m
        dy = self.center_y_m - geometry.center_y_m
        dw = self.width_m - geometry.width_m
        dh = self.height_m - geometry.height_m
        return abs(dx) <= dw and abs(dy) <= dh


class ObjectGeometryProvider(ABC):
    """Interface for classes that provide object geometry."""

    @abstractmethod
    def get_probe_positions(self) -> Sequence[ProbePosition]:
        pass

    @abstractmethod
    def get_object_geometry(self) -> ObjectGeometry:
        pass


class Object:
    """Complex transmission function stored as a (layers, height, width) array with spatial metadata."""

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
            center_x_m=center.coordinate_x_m,
            center_y_m=center.coordinate_y_m,
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


def align_objects(
    reference_object: Object, moving_object: Object, *, upsample_factor: int = 100
) -> Object:
    """Sub-pixel align ``moving_object`` to ``reference_object``.

    Estimates the sub-pixel translation between the two reconstructions with
    ``skimage.registration.phase_cross_correlation`` (run on layer-flattened
    amplitudes), then applies the shift to every layer of the complex
    ``moving_object`` array via a Fourier phase ramp so the complex phase is
    preserved across the interpolation.

    The returned object's ``center`` is offset from the moving object's center
    by ``-shift_yx * pixel_size`` (in meters). This preserves the
    world-coordinate mapping of every probe position that was previously valid
    against ``moving_object``: a probe at world coordinate ``W`` that addressed
    a particular piece of content in ``moving_object`` will, after alignment,
    address that same content at its new array index in the returned object.
    The returned center therefore differs from ``reference_object.get_center()``
    by the alignment shift.

    Args:
        reference_object: The reconstruction whose array indices the result is
            aligned to.
        moving_object: The reconstruction to be re-registered. Must share
            ``reference_object``'s pixel geometry and flattened array shape.
        upsample_factor: Sub-pixel precision passed to
            ``phase_cross_correlation``. Higher values find finer shifts at
            roughly linear cost.
    """
    reference_pixel_geometry = reference_object.get_pixel_geometry()
    moving_pixel_geometry = moving_object.get_pixel_geometry()
    if reference_pixel_geometry != moving_pixel_geometry:
        raise ValueError(
            f'Object pixel geometry mismatch: reference {reference_pixel_geometry} '
            f'vs moving {moving_pixel_geometry}!'
        )

    reference_flat = reference_object.get_layers_flattened()
    moving_flat = moving_object.get_layers_flattened()
    if reference_flat.shape != moving_flat.shape:
        raise ValueError(
            f'Object array shape mismatch: reference {reference_flat.shape} '
            f'vs moving {moving_flat.shape}!'
        )

    shift_yx, _, _ = phase_cross_correlation(
        numpy.absolute(reference_flat),
        numpy.absolute(moving_flat),
        upsample_factor=upsample_factor,
    )
    logger.info(f'align_objects sub-pixel shift (y, x) = {tuple(shift_yx)} px')

    moving_array = moving_object.get_array()
    aligned_array = fourier_shift_2d(moving_array, dx=float(shift_yx[1]), dy=float(shift_yx[0]))

    moving_center = moving_object.get_center()
    new_center = ObjectCenter(
        coordinate_x_m=moving_center.coordinate_x_m
        - float(shift_yx[1]) * moving_pixel_geometry.width_m,
        coordinate_y_m=moving_center.coordinate_y_m
        - float(shift_yx[0]) * moving_pixel_geometry.height_m,
    )

    return Object(
        array=aligned_array,
        pixel_geometry=moving_pixel_geometry.copy(),
        center=new_center,
        layer_spacing_m=list(moving_object.layer_spacing_m),
    )


class ObjectFileReader(ABC):
    """Plugin interface for reading objects."""

    @abstractmethod
    def read(self, file_path: Path) -> Object:
        """Read an object from file."""
        pass


class ObjectFileWriter(ABC):
    """Plugin interface for writing objects."""

    @abstractmethod
    def write(self, file_path: Path, object_: Object) -> None:
        """Write an object to file."""
        pass
