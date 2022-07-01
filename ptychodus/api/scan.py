from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Final, Iterator

import numpy

from .observer import Observable


@dataclass(frozen=True)
class ScanPoint:
    '''scan point coordinates are conventionally in meters'''
    x: Decimal
    y: Decimal


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


class ScanPointSequence(Sequence[ScanPoint]):
    '''a sequence of scan points'''
    pass


class ScanInitializer(ScanPointSequence, Observable):
    '''interface for plugins that can initialize scan sequences'''

    @abstractproperty
    def name(self) -> str:
        '''returns a unique name'''
        pass

    @abstractmethod
    def _getPoint(self, index: int) -> ScanPoint:
        '''returns the scan point'''
        pass

    def __init__(self, rng: numpy.random.Generator) -> None:
        super().__init__()
        self._rng = rng
        self._transform = ScanPointTransform.PXPY
        self._jitterRadiusInMeters = Decimal()

    def getTransform(self) -> ScanPointTransform:
        '''gets the scan point transform'''
        return self._transform

    def setTransform(self, transform: ScanPointTransform) -> None:
        '''sets the scan point transform'''
        if self._transform != transform:
            self._transform = transform
            self.notifyObservers()

    def getJitterRadiusInMeters(self) -> Decimal:
        '''gets the jitter radius'''
        return self._jitterRadiusInMeters

    def setJitterRadiusInMeters(self, jitterRadiusInMeters: Decimal) -> None:
        '''sets the jitter radius'''
        if self._jitterRadiusInMeters != jitterRadiusInMeters:
            self._jitterRadiusInMeters = jitterRadiusInMeters
            self.notifyObservers()

    def __getitem__(self, index: int) -> ScanPoint:
        '''returns the jittered and transformed scan point'''
        point = self._getPoint(index)

        if self._jitterRadiusInMeters > Decimal():
            rad = Decimal(self._rng.uniform())
            dirX = Decimal(self._rng.normal())
            dirY = Decimal(self._rng.normal())

            scalar = self._jitterRadiusInMeters * (rad / (dirX**2 + dirY**2)).sqrt()
            point = ScanPoint(point.x + scalar * dirX, point.y + scalar * dirY)

        return self._transform(point)


class ScanDictionary(Mapping[str, ScanPointSequence]):
    '''a collection of named scan point sequences'''
    pass


class SimpleScanDictionary(ScanDictionary):
    '''a dictionary-based scan file implementation'''
    DEFAULT_SEQUENCE_NAME: Final[str] = 'Default'

    def __init__(self, sequences: Mapping[str, ScanPointSequence]) -> None:
        super().__init__()
        self._sequences = sequences

    @classmethod
    def createFromUnnamedSequence(cls, sequence: ScanPointSequence) -> SimpleScanDictionary:
        '''creates a dictionary containing one unnamed scan sequence'''
        return cls({cls.DEFAULT_SEQUENCE_NAME: sequence})

    def __getitem__(self, key: str) -> ScanPointSequence:
        '''retrieves a scan point sequence from the dictionary'''
        return self._sequences[key]

    def __iter__(self) -> Iterator[str]:
        '''iterates over scan point sequence names'''
        return iter(self._sequences)

    def __len__(self) -> int:
        '''returns the number of scan point sequences in the dictionary'''
        return len(self._sequences)


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
    def read(self, filePath: Path) -> ScanDictionary:
        '''reads a scan dictionary from file'''
        pass


class ScanPointParseError(Exception):
    '''raised when the scan file cannot be parsed'''
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
    def write(self, filePath: Path, scanDict: ScanDictionary) -> None:
        '''writes a scan dictionary to file'''
        pass
