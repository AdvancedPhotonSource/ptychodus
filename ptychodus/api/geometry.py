from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

T = TypeVar('T', int, float, Decimal)


@dataclass(frozen=True)
class PixelGeometry:
    widthInMeters: float
    heightInMeters: float

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.widthInMeters}, {self.heightInMeters})'


@dataclass(frozen=True)
class ImageExtent:
    widthInPixels: int
    heightInPixels: int

    @property
    def size(self) -> int:
        '''returns the number of pixels in the image'''
        return self.widthInPixels * self.heightInPixels

    @property
    def shape(self) -> tuple[int, int]:
        '''returns the image shape (heightInPixels, widthInPixels) tuple'''
        return self.heightInPixels, self.widthInPixels

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageExtent):
            return (self.shape == other.shape)

        return False

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.widthInPixels}, {self.heightInPixels})'


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.x}, {self.y})'


@dataclass(frozen=True)
class Line2D:
    begin: Point2D
    end: Point2D

    def lerp(self, alpha: float) -> Point2D:
        beta = 1 - alpha
        x = beta * self.begin.x + alpha * self.end.x
        y = beta * self.begin.y + alpha * self.end.y
        return Point2D(x, y)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.begin}, {self.end})'


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
    def x_end(self) -> float:
        return self.x + self.width

    @property
    def y_begin(self) -> float:
        return self.y

    @property
    def y_end(self) -> float:
        return self.y + self.height

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.x}, {self.y}, {self.width}, {self.height})'


class Interval(Generic[T]):

    def __init__(self, lower: T, upper: T) -> None:
        self.lower: T = lower
        self.upper: T = upper

    @classmethod
    def createProper(self, a: T, b: T) -> Interval[T]:
        if b < a:
            return Interval[T](b, a)
        else:
            return Interval[T](a, b)

    @property
    def isEmpty(self) -> bool:
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
        return (self.lower <= item and item < self.upper)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.lower}, {self.upper})'
