from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy


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


class PositionSequence(Sequence[ScanPoint]):
    def __init__(self, point_seq: Sequence[ScanPoint] | None = None) -> None:
        indexes: list[int] = []
        coordinates_m: list[float] = []

        if point_seq is not None:
            for point in point_seq:
                indexes.append(point.index)
                coordinates_m.append(point.position_y_m)
                coordinates_m.append(point.position_x_m)

        self._indexes = numpy.array(indexes)
        self._coordinates_m = numpy.reshape(coordinates_m, (-1, 2))

    def copy(self) -> PositionSequence:
        seq = PositionSequence()
        seq._indexes = self._indexes.copy()
        seq._coordinates_m = self._coordinates_m.copy()
        return seq

    @overload
    def __getitem__(self, index: int) -> ScanPoint: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ScanPoint]: ...

    def __getitem__(self, index: int | slice) -> ScanPoint | Sequence[ScanPoint]:
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start, index.stop, index.step)]

        return ScanPoint(
            index=self._indexes[index],
            position_x_m=self._coordinates_m[index, -1],
            position_y_m=self._coordinates_m[index, -2],
        )

    def __len__(self) -> int:
        return self._indexes.size

    @property
    def nbytes(self) -> int:
        return self._indexes.nbytes + self._coordinates_m.nbytes

    def __repr__(self) -> str:
        return f'{self._coordinates_m.dtype}{self._coordinates_m.shape}'


class ScanPointParseError(Exception):
    """raised when the scan file cannot be parsed"""

    pass


class PositionFileReader(ABC):
    """interface for plugins that read position files"""

    @abstractmethod
    def read(self, file_path: Path) -> PositionSequence:
        """reads positions from file"""
        pass


class PositionFileWriter(ABC):
    """interface for plugins that write position files"""

    @abstractmethod
    def write(self, file_path: Path, positions: PositionSequence) -> None:
        """writes positions to file"""
        pass
