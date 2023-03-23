from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar

T = TypeVar('T', int, float, Decimal)


@dataclass(frozen=True)
class Array2D(Generic[T]):
    x: T
    y: T


class Interval(Generic[T]):

    def __init__(self, lower: T, upper: T) -> None:
        self.lower: T = lower
        self.upper: T = upper

    @property
    def isEmpty(self) -> bool:
        return self.upper < self.lower

    def clamp(self, value: T) -> T:
        return max(self.lower, min(value, self.upper))

    def hull(self, value: T) -> Interval[T]:
        return Interval[T](min(self.lower, value), max(self.upper, value))

    @property
    def length(self) -> T:
        return self.upper - self.lower

    @property
    def center(self) -> T:
        fullLength = self.length
        halfLength = fullLength // 2 if isinstance(fullLength, int) else fullLength / 2
        return self.lower + halfLength

    def copy(self) -> Interval[T]:
        return Interval[T](self.lower, self.upper)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.lower}, {self.upper})'


@dataclass(frozen=True)
class Box2D(Generic[T]):
    rangeX: Interval[T]
    rangeY: Interval[T]

    def hull(self, x: T, y: T) -> Box2D[T]:
        return Box2D[T](self.rangeX.hull(x), self.rangeY.hull(y))
