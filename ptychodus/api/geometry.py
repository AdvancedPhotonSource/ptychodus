from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

T = TypeVar('T', int, float, Decimal)


@dataclass(frozen=True)
class Array2D(Generic[T]):  # TODO remove
    x: T
    y: T


@dataclass(frozen=True)
class Point2D(Generic[T]):
    x: T
    y: T


@dataclass(frozen=True)
class Line2D(Generic[T]):
    begin: Point2D[T]
    end: Point2D[T]

    def lerp(self, alpha: T) -> Point2D[T]:
        beta = 1 - alpha
        x = beta * self.begin.x + alpha * self.end.x
        y = beta * self.begin.y + alpha * self.end.y
        return Point2D[T](x, y)


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

    def hull(self, value: T) -> Interval[T]:
        return Interval[T](min(self.lower, value), max(self.upper, value))

    @property
    def width(self) -> T:
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


@dataclass(frozen=True)
class Box2D(Generic[T]):
    rangeX: Interval[T]
    rangeY: Interval[T]

    @property
    def midpoint(self) -> Point2D[T]:
        return Point2D[T](self.rangeX.midrange, self.rangeY.midrange)

    def hull(self, x: T, y: T) -> Box2D[T]:
        return Box2D[T](self.rangeX.hull(x), self.rangeY.hull(y))
