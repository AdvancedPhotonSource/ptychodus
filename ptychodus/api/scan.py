from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload
import sys


@dataclass(frozen=True)
class ScanPoint:
    index: int
    positionXInMeters: float
    positionYInMeters: float


@dataclass(frozen=True)
class ScanBoundingBox:
    minimumXInMeters: float
    maximumXInMeters: float
    minimumYInMeters: float
    maximumYInMeters: float

    @property
    def widthInMeters(self) -> float:
        return self.maximumXInMeters - self.minimumXInMeters

    @property
    def heightInMeters(self) -> float:
        return self.maximumYInMeters - self.minimumYInMeters

    @property
    def centerXInMeters(self) -> float:
        return self.minimumXInMeters + self.widthInMeters / 2.

    @property
    def centerYInMeters(self) -> float:
        return self.minimumYInMeters + self.heightInMeters / 2.

    def hull(self, bbox: ScanBoundingBox) -> ScanBoundingBox:
        return ScanBoundingBox(
            minimumXInMeters=min(self.minimumXInMeters, bbox.minimumXInMeters),
            maximumXInMeters=max(self.maximumXInMeters, bbox.maximumXInMeters),
            minimumYInMeters=min(self.minimumYInMeters, bbox.minimumYInMeters),
            maximumYInMeters=max(self.maximumYInMeters, bbox.maximumYInMeters),
        )


class Scan(Sequence[ScanPoint]):

    def __init__(self, pointSeq: Sequence[ScanPoint] | None = None) -> None:
        self._pointSeq: Sequence[ScanPoint] = [] if pointSeq is None else pointSeq

    def copy(self) -> Scan:
        return Scan(list(self._pointSeq))

    @overload
    def __getitem__(self, index: int) -> ScanPoint:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ScanPoint]:
        ...

    def __getitem__(self, index: int | slice) -> ScanPoint | Sequence[ScanPoint]:
        return self._pointSeq[index]

    def __len__(self) -> int:
        return len(self._pointSeq)

    @property
    def sizeInBytes(self) -> int:
        numBytes = sys.getsizeof(self._pointSeq)
        numBytes += sum(sys.getsizeof(point) for point in self._pointSeq)
        return numBytes


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
