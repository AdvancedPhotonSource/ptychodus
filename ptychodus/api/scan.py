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


class Scan(Sequence[ScanPoint]):

    def __init__(self, pointSeq: Sequence[ScanPoint] | None = None) -> None:
        self._pointSeq: Sequence[ScanPoint] = [] if pointSeq is None else pointSeq

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

    def __sizeof__(self) -> int:
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
