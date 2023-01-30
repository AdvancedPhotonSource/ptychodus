from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any
import logging

import numpy
import numpy.typing

from .observer import Observable

ScanArrayType = numpy.typing.NDArray[numpy.floating[Any]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanPoint:
    '''scan point coordinates are conventionally in meters'''
    x: Decimal
    y: Decimal

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ScanPoint):
            return (self.x == other.x and self.y == other.y)
        else:
            return NotImplemented


class ScanIndexFilter(ABC):
    '''filters scan points by index'''

    @abstractproperty
    def name(self) -> str:
        '''returns a unique name'''
        pass

    @abstractmethod
    def __call__(self, index: int) -> bool:
        '''include scan point if true, remove otherwise'''
        pass


class ScanPointTransform(Enum):
    '''transformations to negate or swap scan point coordinates'''
    PXPY = 0x0
    MXPY = 0x1
    PXMY = 0x2
    MXMY = 0x3
    PYPX = 0x4
    PYMX = 0x5
    MYPX = 0x6
    MYMX = 0x7

    @classmethod
    def fromSimpleName(self, name: str) -> ScanPointTransform:
        transform = ScanPointTransform.PXPY

        try:
            transform = next(xform for xform in ScanPointTransform
                             if name.casefold() == xform.simpleName.casefold())
        except StopIteration:
            logger.debug(f'Invalid ScanPointTransform \"{name}\"')

        return transform

    @property
    def negateX(self) -> bool:
        '''indicates whether the x coordinate is negated'''
        return self.value & 1 != 0

    @property
    def negateY(self) -> bool:
        '''indicates whether the y coordinate is negated'''
        return self.value & 2 != 0

    @property
    def swapXY(self) -> bool:
        '''indicates whether the x and y coordinates are swapped'''
        return self.value & 4 != 0

    @property
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        xp = '-x' if self.negateX else '+x'
        yp = '-y' if self.negateY else '+y'
        return f'{yp}{xp}' if self.swapXY else f'{xp}{yp}'

    @property
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        xp = '\u2212x' if self.negateX else '\u002Bx'
        yp = '\u2212y' if self.negateY else '\u002By'
        return f'({yp}, {xp})' if self.swapXY else f'({xp}, {yp})'

    def __call__(self, point: ScanPoint) -> ScanPoint:
        '''transforms a scan point'''
        xp = -point.x if self.negateX else point.x
        yp = -point.y if self.negateY else point.y
        return ScanPoint(yp, xp) if self.swapXY else ScanPoint(xp, yp)


class Scan(Mapping[int, ScanPoint], Observable):
    '''interface for an enumerated collection of scan points'''

    @abstractproperty
    def name(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass


class TabularScan(Scan):

    def __init__(self, name: str, pointMap: Mapping[int, ScanPoint]) -> None:
        super().__init__()
        self._name = name
        self._data = dict(pointMap)

    @classmethod
    def createEmpty(cls) -> TabularScan:
        return cls('Tabular', {0: ScanPoint(Decimal(), Decimal())})

    @classmethod
    def createFromPointSequence(cls, name: str, pointSeq: Sequence[ScanPoint]) -> TabularScan:
        return cls(name, {index: point for index, point in enumerate(pointSeq)})

    @classmethod
    def createFromMappedPointSequence(
            cls, name: str, pointSeqMap: Mapping[int, Sequence[ScanPoint]]) -> TabularScan:
        pointMap: dict[int, ScanPoint] = dict()

        for index, pointSeq in pointSeqMap.items():
            pointMap[index] = ScanPoint(
                x=median(point.x for point in pointSeq),
                y=median(point.y for point in pointSeq),
            )

        return cls(name, pointMap)

    @property
    def name(self) -> str:
        return self._name

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

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def read(self, filePath: Path) -> Sequence[Scan]:
        '''reads a scan dictionary from file'''
        pass


class ScanFileWriter(ABC):
    '''interface for plugins that write scan files'''

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def write(self, filePath: Path, scanSeq: Sequence[Scan]) -> None:
        '''writes a scan dictionary to file'''
        pass
