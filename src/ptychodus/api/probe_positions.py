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
    x_m: float
    y_m: float
    probe_photon_count: float | None = None


@dataclass(frozen=True)
class ScanGeometry:
    """Bounding box and total path length of a set of probe positions."""

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


class ProbePositionSequence(Sequence[ProbePosition]):
    """Memory-efficient sequence of ProbePosition objects backed by numpy arrays."""

    def __init__(self, point_seq: Sequence[ProbePosition] | None = None) -> None:
        indexes: list[int] = []
        coordinates_m: list[float] = []
        photon_counts: list[float] = []

        if point_seq is not None:
            for point in point_seq:
                indexes.append(point.index)
                coordinates_m.append(point.y_m)
                coordinates_m.append(point.x_m)
                if point.probe_photon_count is not None:
                    photon_counts.append(point.probe_photon_count)

        self._indexes = numpy.array(indexes)
        self._coordinates_m = numpy.reshape(coordinates_m, (-1, 2))
        # All-or-nothing invariant: either every input point supplies probe_photon_count
        # or none of them do. A mix indicates a reader bug; refuse the input so it surfaces
        # immediately rather than silently degrading downstream weighting.
        num_points = len(indexes)
        if len(photon_counts) == 0:
            self._probe_photon_counts: numpy.ndarray | None = None
        elif len(photon_counts) == num_points:
            counts_array = numpy.asarray(photon_counts, dtype=numpy.float64)
            bad = ~numpy.isfinite(counts_array) | (counts_array < 0.0)
            if bad.any():
                first = int(numpy.argmax(bad))
                raise ValueError(
                    'ProbePositionSequence requires finite non-negative probe_photon_count; '
                    f'got {counts_array[first]!r} at point index {indexes[first]}.'
                )
            self._probe_photon_counts = counts_array
        else:
            raise ValueError(
                'ProbePositionSequence requires probe_photon_count on every point or none; '
                f'got {len(photon_counts)} populated out of {num_points}.'
            )

    def copy(self) -> ProbePositionSequence:
        seq = ProbePositionSequence()
        seq._indexes = self._indexes.copy()
        seq._coordinates_m = self._coordinates_m.copy()
        seq._probe_photon_counts = (
            None if self._probe_photon_counts is None else self._probe_photon_counts.copy()
        )
        return seq

    @overload
    def __getitem__(self, index: int) -> ProbePosition: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProbePosition]: ...

    def __getitem__(self, index: int | slice) -> ProbePosition | Sequence[ProbePosition]:
        if isinstance(index, slice):
            # Slice the backing arrays directly rather than materializing a
            # ProbePosition per element. Basic numpy slicing returns views; that
            # is safe because this class exposes no mutators.
            seq = ProbePositionSequence()
            seq._indexes = self._indexes[index]
            seq._coordinates_m = self._coordinates_m[index, :]
            seq._probe_photon_counts = (
                None if self._probe_photon_counts is None else self._probe_photon_counts[index]
            )
            return seq

        photon_count: float | None = None
        if self._probe_photon_counts is not None:
            photon_count = float(self._probe_photon_counts[index])

        return ProbePosition(
            index=self._indexes[index],
            x_m=self._coordinates_m[index, -1],
            y_m=self._coordinates_m[index, -2],
            probe_photon_count=photon_count,
        )

    def __len__(self) -> int:
        return self._indexes.size

    def get_probe_photon_counts(self) -> numpy.ndarray | None:
        """Return the per-position photon-count array, or ``None`` when the reader did not measure it.

        The all-or-nothing invariant guaranteed at construction means the returned array,
        when non-None, is fully populated and 1:1 aligned with the index / coordinate arrays.
        """
        return self._probe_photon_counts

    @property
    def nbytes(self) -> int:
        total = self._indexes.nbytes + self._coordinates_m.nbytes
        if self._probe_photon_counts is not None:
            total += self._probe_photon_counts.nbytes
        return total

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


def calculate_scan_geometry(positions: Iterable[ProbePosition]) -> ScanGeometry | None:
    """Compute the bounding box and total path length of a set of probe positions; returns None if empty.

    ``positions`` is iterated twice (once for the bounding box, once for the
    path length), so it must be a re-iterable sequence. Passing a one-shot
    generator will silently report ``length_m=0``.
    """
    minimum_x_m = +numpy.inf
    maximum_x_m = -numpy.inf
    minimum_y_m = +numpy.inf
    maximum_y_m = -numpy.inf
    length_m = 0.0

    for point in positions:
        if point.x_m < minimum_x_m:
            minimum_x_m = point.x_m

        if maximum_x_m < point.x_m:
            maximum_x_m = point.x_m

        if point.y_m < minimum_y_m:
            minimum_y_m = point.y_m

        if maximum_y_m < point.y_m:
            maximum_y_m = point.y_m

    is_empty_x = maximum_x_m < minimum_x_m
    is_empty_y = maximum_y_m < minimum_y_m

    if is_empty_x or is_empty_y:
        return None

    for point_l, point_r in pairwise(positions):
        dx = point_r.x_m - point_l.x_m
        dy = point_r.y_m - point_l.y_m
        length_m += numpy.hypot(dx, dy)

    return ScanGeometry(
        minimum_x_m=minimum_x_m,
        maximum_x_m=maximum_x_m,
        minimum_y_m=minimum_y_m,
        maximum_y_m=maximum_y_m,
        length_m=length_m,
    )
