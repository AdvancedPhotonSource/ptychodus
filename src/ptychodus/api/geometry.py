from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

T = TypeVar('T', int, float, Decimal)


@dataclass(frozen=True)
class AffineTransform:
    a00: float
    a01: float
    a02: float

    a10: float
    a11: float
    a12: float

    def __call__(self, y: float, x: float) -> tuple[float, float]:
        yp = self.a00 * y + self.a01 * x + self.a02
        xp = self.a10 * y + self.a11 * x + self.a12
        return yp, xp


@dataclass(frozen=True)
class PixelGeometry:
    width_m: float
    height_m: float

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def aspect_ratio(self) -> float:
        return self.width_m / self.height_m

    def copy(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=float(self.width_m),
            height_m=float(self.height_m),
        )


@dataclass(frozen=True)
class ImageExtent:
    width_px: int
    height_px: int

    @property
    def size(self) -> int:
        """returns the number of pixels in the image"""
        return self.width_px * self.height_px

    @property
    def shape(self) -> tuple[int, int]:
        """returns the image shape (height_px, width_px) tuple"""
        return self.height_px, self.width_px

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageExtent):
            return self.shape == other.shape

        return False


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Line2D:
    begin: Point2D
    end: Point2D

    def lerp(self, alpha: float) -> Point2D:
        beta = 1 - alpha
        x = beta * self.begin.x + alpha * self.end.x
        y = beta * self.begin.y + alpha * self.end.y
        return Point2D(x, y)


@dataclass(frozen=True)
class Box2D:
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
    def __init__(self, lower: T, upper: T) -> None:
        self.lower: T = lower
        self.upper: T = upper

    @classmethod
    def create_proper(cls, a: T, b: T) -> Interval[T]:
        if b < a:
            return Interval[T](b, a)
        else:
            return Interval[T](a, b)

    @property
    def is_empty(self) -> bool:
        return self.upper < self.lower

    def clamp(self, value: T) -> T:
        return max(self.lower, min(value, self.upper))

    def hull(self, value: Interval[T] | T) -> Interval[T]:
        if isinstance(value, Interval):
            return Interval[T](min(self.lower, value.lower), max(self.upper, value.upper))
        else:
            return Interval[T](min(self.lower, value), max(self.upper, value))

    @property
    def length(self) -> T:
        return self.upper - self.lower

    @property
    def midrange(self) -> T:
        total = self.lower + self.upper
        return total // 2 if isinstance(total, int) else total / 2

    def copy(self) -> Interval[T]:
        return Interval[T](self.lower, self.upper)

    def __contains__(self, item: T) -> bool:
        return self.lower <= item and item < self.upper

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.lower}, {self.upper})'
