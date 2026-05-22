"""2D geometric primitives: points, lines, boxes, intervals, and pixel/image geometry."""

from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

import numpy
import scipy.special

from .common import ComplexArrayType, RealArrayType

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
    """Closed interval [lower, upper] with clamp and membership operations."""

    def __init__(self, lower: T, upper: T) -> None:
        self.lower: T = lower
        self.upper: T = upper

    @classmethod
    def create_proper(cls, a: T, b: T) -> Interval[T]:
        if b < a:
            return Interval[T](b, a)
        else:
            return Interval[T](a, b)

    def clamp(self, value: T) -> T:
        return max(self.lower, min(value, self.upper))

    def __contains__(self, item: T) -> bool:
        return self.lower <= item <= self.upper

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.lower}, {self.upper})'


@dataclass(frozen=True)
class ZernikeMode:
    """A single Zernike polynomial term with a complex coefficient, radial degree n, and angular frequency m."""

    coefficient: complex
    radial_degree: int  # n
    angular_frequency: int  # m

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
    ) -> ComplexArrayType:
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


@dataclass(frozen=True)
class HermiteMode:
    """A single 2D Hermite polynomial term H_m(x) * H_n(y) with a complex coefficient and non-negative orders m (x) and n (y)."""

    coefficient: complex
    order_x: int  # m
    order_y: int  # n

    def _hermite(self, order: int, value: RealArrayType) -> RealArrayType:
        return scipy.special.eval_hermite(order, value)

    def __call__(self, x: RealArrayType, y: RealArrayType) -> ComplexArrayType:
        hx = self._hermite(self.order_x, x)
        hy = self._hermite(self.order_y, y)
        return self.coefficient * hx * hy

    def __str__(self) -> str:
        return f'{self.coefficient}$H_{{{self.order_x},{self.order_y}}}(x,y)$'


def fourier_gradient(
    image: ComplexArrayType, pixel_geometry: PixelGeometry | None = None
) -> tuple[ComplexArrayType, ComplexArrayType]:
    """Calculate the Fourier-differentiation gradient of an image.

    If ``pixel_geometry`` is provided, the returned gradient is in units of
    ``image_units / m``; otherwise it is in ``image_units / pixel``.
    """
    dy = pixel_geometry.height_m if pixel_geometry is not None else 1.0
    dx = pixel_geometry.width_m if pixel_geometry is not None else 1.0

    u = numpy.fft.fftfreq(image.shape[-2], d=dy).reshape(-1, 1)
    v = numpy.fft.fftfreq(image.shape[-1], d=dx)

    grad_y = numpy.fft.ifft(numpy.fft.fft(image, axis=-2) * (2j * numpy.pi * u), axis=-2)
    grad_x = numpy.fft.ifft(numpy.fft.fft(image, axis=-1) * (2j * numpy.pi * v), axis=-1)

    return grad_y, grad_x
