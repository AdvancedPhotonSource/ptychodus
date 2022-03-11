from __future__ import annotations
from typing import Generic, Iterable, overload, TypeVar, Union
from decimal import Decimal

T = TypeVar('T', int, float, Decimal)


class Interval(Generic[T]):
    def __init__(self, lower: T, upper: T) -> None:
        self.lower: T = lower
        self.upper: T = upper

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

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.lower}, {self.upper})'


class Box(Generic[T]):
    def __init__(self, intervals: Iterable[Interval[T]]) -> None:
        self._intervalList: list[Interval[T]] = [x for x in intervals]

    def __iter__(self):
        return iter(self._intervalList)

    @overload
    def __getitem__(self, idx: int) -> Interval[T]:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Box[T]:
        ...

    def __getitem__(self, idx: Union[slice, int]) -> Union[Box[T], Interval[T]]:
        if isinstance(idx, slice):
            return Box(self._intervalList[idx])
        else:
            return self._intervalList[idx]

    def __len__(self) -> int:
        return len(self._intervalList)
