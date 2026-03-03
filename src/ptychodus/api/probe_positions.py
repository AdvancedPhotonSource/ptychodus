"""Probe position (scan point) data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import overload

import numpy


@dataclass(frozen=True)
class ProbePosition:
    """Probe position with its scan index and (x, y) physical coordinates in meters."""

    index: int
    coordinate_x_m: float
    coordinate_y_m: float


@dataclass(frozen=True)
class ScanGeometry:
    minimum_x_m: float = +numpy.inf
    maximum_x_m: float = -numpy.inf
    minimum_y_m: float = +numpy.inf
    maximum_y_m: float = -numpy.inf
    length_m: float = 0.0

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


def calculate_scan_geometry(positions: Iterable[ProbePosition]) -> ScanGeometry | None:
    minimum_x_m = +numpy.inf
    maximum_x_m = -numpy.inf
    minimum_y_m = +numpy.inf
    maximum_y_m = -numpy.inf
    length_m = 0.0

    for point in positions:
        if point.coordinate_x_m < minimum_x_m:
            minimum_x_m = point.coordinate_x_m

        if maximum_x_m < point.coordinate_x_m:
            maximum_x_m = point.coordinate_x_m

        if point.coordinate_y_m < minimum_y_m:
            minimum_y_m = point.coordinate_y_m

        if maximum_y_m < point.coordinate_y_m:
            maximum_y_m = point.coordinate_y_m

    is_empty_x = maximum_x_m < minimum_x_m
    is_empty_y = maximum_y_m < minimum_y_m

    if is_empty_x or is_empty_y:
        return None

    for point_l, point_r in pairwise(positions):
        dx = point_r.coordinate_x_m - point_l.coordinate_x_m
        dy = point_r.coordinate_y_m - point_l.coordinate_y_m
        length_m += numpy.hypot(dx, dy)

    return ScanGeometry(
        minimum_x_m=minimum_x_m,
        maximum_x_m=maximum_x_m,
        minimum_y_m=minimum_y_m,
        maximum_y_m=maximum_y_m,
        length_m=length_m,
    )


class ProbePositionSequence(Sequence[ProbePosition]):
    """Memory-efficient sequence of ProbePosition objects backed by numpy arrays."""

    def __init__(self, point_seq: Sequence[ProbePosition] | None = None) -> None:
        indexes: list[int] = []
        coordinates_m: list[float] = []

        if point_seq is not None:
            for point in point_seq:
                indexes.append(point.index)
                coordinates_m.append(point.coordinate_y_m)
                coordinates_m.append(point.coordinate_x_m)

        self._indexes = numpy.array(indexes)
        self._coordinates_m = numpy.reshape(coordinates_m, (-1, 2))

    def copy(self) -> ProbePositionSequence:
        seq = ProbePositionSequence()
        seq._indexes = self._indexes.copy()
        seq._coordinates_m = self._coordinates_m.copy()
        return seq

    @overload
    def __getitem__(self, index: int) -> ProbePosition: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProbePosition]: ...

    def __getitem__(self, index: int | slice) -> ProbePosition | Sequence[ProbePosition]:
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start, index.stop, index.step)]

        return ProbePosition(
            index=self._indexes[index],
            coordinate_x_m=self._coordinates_m[index, -1],
            coordinate_y_m=self._coordinates_m[index, -2],
        )

    def __len__(self) -> int:
        return self._indexes.size

    @property
    def nbytes(self) -> int:
        return self._indexes.nbytes + self._coordinates_m.nbytes

    def __repr__(self) -> str:
        return f'{self._coordinates_m.dtype}{self._coordinates_m.shape}'


class ProbePositionParseError(Exception):
    """Raised when a probe position file cannot be parsed."""

    pass


class ProbePositionFileReader(ABC):
    """Plugin interface for reading probe position files."""

    @abstractmethod
    def read(self, file_path: Path) -> ProbePositionSequence:
        """Read probe positions from file."""
        pass


class ProbePositionFileWriter(ABC):
    """Plugin interface for writing probe position files."""

    @abstractmethod
    def write(self, file_path: Path, positions: ProbePositionSequence) -> None:
        """Write probe positions to file."""
        pass
