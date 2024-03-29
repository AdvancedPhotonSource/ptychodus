from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from enum import auto, Enum
from pathlib import Path
from statistics import median
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .geometry import Point2D

CoordinateArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ScanIndexes = numpy.typing.NDArray[numpy.integer[Any]]


class ScanIndexFilter(Enum):
    '''filters scan points by index'''
    ALL = auto()
    ODD = auto()
    EVEN = auto()

    @property
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        return self.name

    @property
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        return self.name.title()

    def __call__(self, index: int) -> bool:
        '''include scan point if true, exclude otherwise'''
        if self is ScanIndexFilter.ODD:
            return (index % 2 != 0)
        elif self is ScanIndexFilter.EVEN:
            return (index % 2 == 0)

        return True


# scan point coordinates are conventionally in meters
ScanPoint: TypeAlias = Point2D[float]
Scan: TypeAlias = Mapping[int, ScanPoint]


class TabularScan(Scan):

    def __init__(self, pointMap: Mapping[int, ScanPoint]) -> None:
        super().__init__()
        self._data = dict(pointMap)

    @classmethod
    def createFromPointIterable(cls, pointIterable: Iterable[ScanPoint]) -> TabularScan:
        return cls({index: point for index, point in enumerate(pointIterable)})

    @classmethod
    def createFromMappedPointIterable(
            cls, pointIterableMap: Mapping[int, Iterable[ScanPoint]]) -> TabularScan:
        pointMap: dict[int, ScanPoint] = dict()

        for index, pointIterable in pointIterableMap.items():
            pointMap[index] = ScanPoint(
                x=median(point.x for point in pointIterable),
                y=median(point.y for point in pointIterable),
            )

        return cls(pointMap)

    def __iter__(self) -> Iterator[int]:
        return iter(self._data)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)


class ScanPointParseError(Exception):
    '''raised when the scan file cannot be parsed'''
    pass


class ScanFileReader(ABC):
    '''interface for plugins that read scan files'''

    @abstractmethod
    def read(self, filePath: Path) -> Scan:
        '''reads a scan dictionary from file'''
        pass


class ScanFileWriter(ABC):
    '''interface for plugins that write scan files'''

    @abstractmethod
    def write(self, filePath: Path, scan: Scan) -> None:
        '''writes a scan dictionary to file'''
        pass
