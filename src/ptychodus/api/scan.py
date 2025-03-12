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
    position_x_m: float
    position_y_m: float


@dataclass(frozen=True)
class ScanBoundingBox:
    minimum_x_m: float
    maximum_x_m: float
    minimum_y_m: float
    maximum_y_m: float

    @property
    def width_m(self) -> float:
        return self.maximum_x_m - self.minimum_x_m

    @property
    def height_m(self) -> float:
        return self.maximum_y_m - self.minimum_y_m

    @property
    def center_x_m(self) -> float:
        return self.minimum_x_m + self.width_m / 2.0

    @property
    def center_y_m(self) -> float:
        return self.minimum_y_m + self.height_m / 2.0

    def hull(self, bbox: ScanBoundingBox) -> ScanBoundingBox:
        return ScanBoundingBox(
            minimum_x_m=min(self.minimum_x_m, bbox.minimum_x_m),
            maximum_x_m=max(self.maximum_x_m, bbox.maximum_x_m),
            minimum_y_m=min(self.minimum_y_m, bbox.minimum_y_m),
            maximum_y_m=max(self.maximum_y_m, bbox.maximum_y_m),
        )


class Scan(Sequence[ScanPoint]):
    def __init__(self, point_seq: Sequence[ScanPoint] | None = None) -> None:
        self._pointSeq: Sequence[ScanPoint] = [] if point_seq is None else point_seq

    def copy(self) -> Scan:
        return Scan(list(self._pointSeq))

    @overload
    def __getitem__(self, index: int) -> ScanPoint: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ScanPoint]: ...

    def __getitem__(self, index: int | slice) -> ScanPoint | Sequence[ScanPoint]:
        return self._pointSeq[index]

    def __len__(self) -> int:
        return len(self._pointSeq)

    @property
    def nbytes(self) -> int:
        sz = sys.getsizeof(self._pointSeq)
        sz += sum(sys.getsizeof(point) for point in self._pointSeq)
        return sz


class ScanPointParseError(Exception):
    """raised when the scan file cannot be parsed"""

    pass


class ScanFileReader(ABC):
    """interface for plugins that read scan files"""

    @abstractmethod
    def read(self, file_path: Path) -> Scan:
        """reads a scan dictionary from file"""
        pass


class ScanFileWriter(ABC):
    """interface for plugins that write scan files"""

    @abstractmethod
    def write(self, file_path: Path, scan: Scan) -> None:
        """writes a scan dictionary to file"""
        pass
