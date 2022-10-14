from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, Iterable, Iterator, overload, TypeVar, Union

T = TypeVar('T', int, float, Decimal)


@dataclass(frozen=True)
class Vector2D(Generic[T]):
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

    def hull(self, value: T) -> None:
        if value < self.lower:
            self.lower = value

        if value > self.upper:
            self.upper = value

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


class Box(Sequence[Interval[T]]):

    def __init__(self, intervals: Iterable[Interval[T]]) -> None:
        self._intervalList: list[Interval[T]] = [x for x in intervals]

    @overload
    def __getitem__(self, index: int) -> Interval[T]:
        ...

    @overload
    def __getitem__(self, index: slice) -> Box[T]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Interval[T], Box[T]]:
        if isinstance(index, slice):
            return Box(self._intervalList[index])
        else:
            return self._intervalList[index]

    def __len__(self) -> int:
        return len(self._intervalList)
