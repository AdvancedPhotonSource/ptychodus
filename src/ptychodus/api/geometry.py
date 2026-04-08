"""2D geometric primitives: points, lines, boxes, intervals, and pixel/image geometry."""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

import numpy
import scipy.special

from .common import RealArrayType

T = TypeVar('T', int, float, Decimal)


@dataclass(frozen=True)
class AffineTransform:
    """2D affine transformation expressed as a 2x3 matrix; callable as transform(x, y) -> (x', y')."""

    a00: float
    a01: float
    a02: float

    a10: float
    a11: float
    a12: float

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        xp = self.a00 * x + self.a01 * y + self.a02
        yp = self.a10 * x + self.a11 * y + self.a12
        return xp, yp


@dataclass(frozen=True)
class PixelGeometry:
    """Physical dimensions of a single detector or probe pixel in meters."""

    width_m: float
    height_m: float

    @property
    def is_square(self) -> bool:
        return self.width_m == self.height_m

    def get_area_m2(self) -> float:
        return self.width_m * self.height_m

    def get_aspect_ratio(self) -> float:
        return self.width_m / self.height_m

    def copy(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=float(self.width_m),
            height_m=float(self.height_m),
        )


@dataclass(frozen=True)
class ImageExtent:
    """Integer pixel dimensions (width × height) of an image or detector."""

    width_px: int
    height_px: int

    def get_size(self) -> int:
        """Return the number of pixels in the image."""
        return self.width_px * self.height_px

    def get_shape(self) -> tuple[int, int]:
        """Return the image shape as a (height_px, width_px) tuple."""
        return self.height_px, self.width_px

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageExtent):
            return self.get_shape() == other.get_shape()

        return False


@dataclass(frozen=True)
class Point2D:
    """Floating-point 2D point."""

    x: float
    y: float


@dataclass(frozen=True)
class Line2D:
    """2D line segment between two points with linear interpolation."""

    begin: Point2D
    end: Point2D

    def lerp(self, alpha: float) -> Point2D:
        beta = 1 - alpha
        x = beta * self.begin.x + alpha * self.end.x
        y = beta * self.begin.y + alpha * self.end.y
        return Point2D(x, y)


@dataclass(frozen=True)
class Box2D:
    """Axis-aligned 2D bounding box defined by its top-left corner, width, and height."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x_begin(self) -> float:
        return self.x

    @property
    def x_center(self) -> float:
        return self.x + self.width / 2

    @property
    def x_end(self) -> float:
        return self.x + self.width

    @property
    def y_begin(self) -> float:
        return self.y

    @property
    def y_center(self) -> float:
        return self.y + self.height / 2

    @property
    def y_end(self) -> float:
        return self.y + self.height


class Interval(Generic[T]):
    """Closed interval [lower, upper] with clamp, hull, and membership operations."""

    def __init__(self, lower: T, upper: T) -> None:
        self.lower: T = lower
        self.upper: T = upper

    @classmethod
    def create_proper(cls, a: T, b: T) -> Interval[T]:
        if b < a:
            return Interval[T](b, a)
        else:
            return Interval[T](a, b)

    def is_empty(self) -> bool:
        return self.upper < self.lower

    def clamp(self, value: T) -> T:
        return max(self.lower, min(value, self.upper))

    def hull(self, value: Interval[T] | T) -> Interval[T]:
        if isinstance(value, Interval):
            return Interval[T](min(self.lower, value.lower), max(self.upper, value.upper))
        else:
            return Interval[T](min(self.lower, value), max(self.upper, value))

    def get_length(self) -> T:
        return self.upper - self.lower

    def get_midrange(self) -> T:
        total = self.lower + self.upper
        return total // 2 if isinstance(total, int) else total / 2

    def copy(self) -> Interval[T]:
        return Interval[T](self.lower, self.upper)

    def __contains__(self, item: T) -> bool:
        return self.lower <= item and item < self.upper

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.lower}, {self.upper})'


@dataclass(frozen=True)
class ZernikeMonomial:
    """A single Zernike polynomial term with a complex coefficient, radial degree n, and angular frequency m."""

    coefficient: complex
    radial_degree: int  # n
    angular_frequency: int  # m

    @property
    def spatial_frequencey(self) -> int:
        return self.radial_degree + abs(self.angular_frequency)

    def _radial_polynomial(self, distance: RealArrayType) -> RealArrayType:
        n_minus_m = self.radial_degree - abs(self.angular_frequency)
        half_n_minus_m = n_minus_m // 2
        sgn = 1

        values = numpy.zeros_like(distance)

        for k in range(half_n_minus_m + 1):
            n_minus_k = self.radial_degree - k
            n_minus_2k = self.radial_degree - 2 * k

            coef = sgn
            coef *= scipy.special.binom(n_minus_k, k)
            coef *= scipy.special.binom(n_minus_2k, half_n_minus_m - k)
            coef = int(coef)  # NOTE!

            values += numpy.multiply(coef, numpy.power(distance, n_minus_2k))

            sgn = -sgn

        return values

    def _angular_function(self, angle: RealArrayType) -> RealArrayType:
        return (
            numpy.sin(-self.angular_frequency * angle)
            if self.angular_frequency < 0
            else numpy.cos(self.angular_frequency * angle)
        )

    def __call__(
        self, distance: RealArrayType, angle: RealArrayType, undefined_value: complex = 0j
    ) -> RealArrayType:
        rvalue = self._radial_polynomial(distance)
        avalue = self._angular_function(angle)
        nvalue_sq = self.radial_degree + 1

        if self.angular_frequency != 0:
            nvalue_sq *= 2

        return numpy.where(
            numpy.logical_and(0 < distance, distance <= 1),
            self.coefficient * numpy.sqrt(nvalue_sq) * rvalue * avalue,
            undefined_value,
        )

    def __str__(self) -> str:
        return f'{self.coefficient}$Z_{{{self.radial_degree}}}^{{{self.angular_frequency:+d}}}$'
