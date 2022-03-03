from __future__ import annotations
from typing import Generic, Iterable, overload, TypeVar, Union
from decimal import Decimal

T = TypeVar('T', int, float, Decimal)


class Interval(Generic[T]):
    def __init__(self, xmin: T, xmax: T) -> None:
        self.xmin: T = xmin
        self.xmax: T = xmax

    def hull(self, value: T) -> None:
        if value < self.xmin:
            self.xmin = value

        if value > self.xmax:
            self.xmax = value

    @property
    def length(self) -> T:
        return self.xmax - self.xmin

    @property
    def center(self) -> T:
        return self.xmin + self.length / 2

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.xmin}, {self.xmax})'


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
