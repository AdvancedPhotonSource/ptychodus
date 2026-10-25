"""Online accumulation of the per-pixel and per-pattern statistics of a diffraction stream.

One pass over the patterns, in chunks of any size and in any order, gathers everything
a coarse estimate of the probe and the object needs: the running mean, minimum and
maximum frames, and each pattern's total and intensity centroid keyed by its scan index.
Every per-pattern quantity depends only on its own frame, so nothing has to be seen
twice and partial results from separate workers merge exactly.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, replace
import logging
import threading
from typing import cast

import numpy

from .assemble import DetectorCorrections, estimate_incident_counts
from .diffraction import (
    BadPixels,
    DetectorBackground,
    DiffractionIndexes,
    DiffractionPattern,
    DiffractionPatterns,
    FlatField,
)
from .moments import FrameStatistics, compute_intensity_moments
from .preprocess.diffraction import inpaint_bad_pixels
from .typing import IntegerArrayType, RealArrayType

__all__ = [
    'DiffractionMomentAccumulator',
    'DiffractionMoments',
    'compute_diffraction_moments',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiffractionMoments:
    """A snapshot of accumulated diffraction statistics; frozen, and picklable.

    The frames are those of the patterns as they were added, before bad pixels were
    filled; `bad_pixels` is the mask the accumulator inpainted with. Per-pattern rows
    pair with probe positions by `indexes`, never by order. Totals sum the whole frame
    with its bad pixels inpainted. Centroids are in detector pixels about the frame
    midpoint ``(N - 1) / 2``, x along columns and y along rows; a frame with no signal
    has a NaN centroid.
    """

    num_patterns: int
    minimum_frame: DiffractionPattern
    mean_frame: RealArrayType
    maximum_frame: DiffractionPattern
    bad_pixels: BadPixels
    indexes: IntegerArrayType
    total_counts: RealArrayType
    center_of_mass_x_px: RealArrayType
    center_of_mass_y_px: RealArrayType

    def mean_pattern(self) -> RealArrayType:
        """The mean frame with its bad pixels inpainted.

        Inpainting is linear for a fixed mask, so this equals the mean of the inpainted
        frames exactly, while costing one fill rather than one per pattern.
        """
        # A float64 frame inpaints to float64; the overload only promises the dtype family.
        return cast(RealArrayType, inpaint_bad_pixels(self.mean_frame, self.bad_pixels))

    def select(self, indexes: Sequence[int] | IntegerArrayType) -> DiffractionMoments:
        """Keep the per-pattern rows whose index is in *indexes*, in their stored order.

        The frames stay those of every pattern accumulated: the mean pattern describes
        the probe, not any one position, so a subset of positions does not change it.
        """
        keep = numpy.isin(self.indexes, numpy.asarray(indexes))
        return replace(
            self,
            indexes=self.indexes[keep],
            total_counts=self.total_counts[keep],
            center_of_mass_x_px=self.center_of_mass_x_px[keep],
            center_of_mass_y_px=self.center_of_mass_y_px[keep],
        )

    def normalize_by_monitor(
        self, monitor_indexes: IntegerArrayType, monitor_counts: RealArrayType
    ) -> DiffractionMoments:
        """Scale each total by ``mean(M) / M_j`` so a change in the beam does not read as absorption.

        *monitor_counts* are measured incident counts, such as an upstream ion chamber
        records, keyed by *monitor_indexes*; each must be unique. A count that is not a
        positive finite number cannot normalize anything and leaves its total unscaled,
        with a warning, as does a pattern whose index has no count.
        """
        unique_indexes, first = numpy.unique(monitor_indexes, return_index=True)

        if unique_indexes.size != monitor_indexes.size:
            raise ValueError('Monitor indexes must be unique!')

        counts = numpy.asarray(monitor_counts, dtype=numpy.float64)[first]
        position = numpy.clip(numpy.searchsorted(unique_indexes, self.indexes), 0, None)
        found = (position < unique_indexes.size) & (
            unique_indexes[numpy.minimum(position, unique_indexes.size - 1)] == self.indexes
        )
        monitor = numpy.full(self.indexes.shape, numpy.nan)
        monitor[found] = counts[position[found]]
        usable = numpy.isfinite(monitor) & (monitor > 0.0)
        total_counts = self.total_counts.copy()

        if usable.any():
            total_counts[usable] *= numpy.mean(monitor[usable]) / monitor[usable]
            logger.info('Normalizing totals by the measured incident photon counts.')

        if not usable.all():
            logger.warning(
                f'{int(numpy.count_nonzero(~usable))} incident photon counts are missing or'
                ' not positive; leaving those totals unnormalized.'
            )

        return replace(self, total_counts=total_counts)

    def estimate_incident_counts(self, *, quantile: float = 0.99) -> float:
        """The unattenuated incident count; see :func:`estimate_incident_counts`."""
        return estimate_incident_counts(self.total_counts, quantile=quantile)


class DiffractionMomentAccumulator:
    """Thread-safe online accumulation of the statistics a :class:`DiffractionMoments` holds.

    Patterns are added already corrected for any detector background and flat field,
    in chunks of any size and from any number of threads. They are buffered into chunks
    of *chunk_size* so the biharmonic fill of *bad_pixels* is solved once per chunk
    rather than once per frame; each full chunk is reduced outside the lock and folded
    in under it.

    `merge` folds in another accumulator or a snapshot, which is how workers that saw
    different patterns -- separate processes included, since snapshots pickle --
    combine into one result equal to a single pass over all of them.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        bad_pixels: BadPixels | None = None,
        chunk_size: int = 256,
    ) -> None:
        if chunk_size < 1:
            raise ValueError('Chunks must hold at least one pattern!')

        self._shape = (height, width)
        self._bad_pixels: BadPixels = (
            numpy.zeros(self._shape, dtype=numpy.bool) if bad_pixels is None else bad_pixels
        )

        if self._bad_pixels.shape != self._shape:
            raise ValueError(
                f'Bad pixels of shape {self._bad_pixels.shape} do not match the {self._shape} '
                'frames!'
            )

        self._chunk_size = chunk_size
        self._frame_statistics = FrameStatistics(height, width)
        self._lock = threading.Lock()
        self._pending_indexes: list[IntegerArrayType] = []
        self._pending_patterns: list[DiffractionPatterns] = []
        self._num_pending = 0
        self._indexes: list[IntegerArrayType] = []
        self._total_counts: list[RealArrayType] = []
        self._center_of_mass_x_px: list[RealArrayType] = []
        self._center_of_mass_y_px: list[RealArrayType] = []

    def add(self, indexes: DiffractionIndexes, patterns: DiffractionPatterns) -> None:
        """Buffer patterns, keyed by their scan indexes, and reduce every chunk that fills."""
        if patterns.shape[-2:] != self._shape:
            raise ValueError(
                f'Patterns of shape {patterns.shape[-2:]} do not match the {self._shape} frames!'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Expected one index per pattern!')

        with self._lock:
            self._pending_indexes.append(numpy.asarray(indexes))
            self._pending_patterns.append(patterns)
            self._num_pending += patterns.shape[0]
            chunk = self._take_pending() if self._num_pending >= self._chunk_size else None

        if chunk is not None:
            self._reduce(*chunk)

    def merge(self, other: DiffractionMomentAccumulator | DiffractionMoments) -> None:
        """Fold in another accumulator, or a snapshot of one, as if its patterns were added here."""
        moments = other.get_moments() if isinstance(other, DiffractionMomentAccumulator) else other

        if moments is None:
            return

        if not numpy.array_equal(moments.bad_pixels, self._bad_pixels):
            raise ValueError('Only accumulations with the same bad pixels can be merged!')

        self._frame_statistics.merge(
            FrameStatistics.from_frames(
                moments.num_patterns,
                moments.minimum_frame,
                moments.mean_frame,
                moments.maximum_frame,
            )
        )
        self._fold_rows(
            moments.indexes,
            moments.total_counts,
            moments.center_of_mass_x_px,
            moments.center_of_mass_y_px,
        )

    def get_moments(self) -> DiffractionMoments | None:
        """Reduce any partial chunk and snapshot the statistics, or None if nothing was added.

        Accumulation may continue afterwards; the snapshot does not change with it.
        """
        with self._lock:
            chunk = self._take_pending() if self._num_pending > 0 else None

        if chunk is not None:
            self._reduce(*chunk)

        frames = self._frame_statistics.get_frames()

        if frames is None:
            return None

        minimum_frame, mean_frame, maximum_frame = frames

        with self._lock:
            return DiffractionMoments(
                num_patterns=self._frame_statistics.num_patterns,
                minimum_frame=minimum_frame,
                mean_frame=mean_frame,
                maximum_frame=maximum_frame,
                bad_pixels=self._bad_pixels.copy(),
                indexes=numpy.concatenate(self._indexes),
                total_counts=numpy.concatenate(self._total_counts),
                center_of_mass_x_px=numpy.concatenate(self._center_of_mass_x_px),
                center_of_mass_y_px=numpy.concatenate(self._center_of_mass_y_px),
            )

    def _take_pending(self) -> tuple[IntegerArrayType, DiffractionPatterns]:
        # Called with the lock held.
        indexes = numpy.concatenate(self._pending_indexes)
        patterns = numpy.concatenate(self._pending_patterns)
        self._pending_indexes.clear()
        self._pending_patterns.clear()
        self._num_pending = 0
        return indexes, patterns

    def _reduce(self, indexes: IntegerArrayType, patterns: DiffractionPatterns) -> None:
        self._frame_statistics.add(patterns)
        repaired = inpaint_bad_pixels(patterns.astype(numpy.float64), self._bad_pixels)
        moments = compute_intensity_moments(repaired)
        self._fold_rows(indexes, moments.total, moments.centroid_x_px, moments.centroid_y_px)

    def _fold_rows(
        self,
        indexes: IntegerArrayType,
        total_counts: RealArrayType,
        center_of_mass_x_px: RealArrayType,
        center_of_mass_y_px: RealArrayType,
    ) -> None:
        with self._lock:
            self._indexes.append(numpy.asarray(indexes).copy())
            self._total_counts.append(numpy.asarray(total_counts, dtype=numpy.float64).copy())
            self._center_of_mass_x_px.append(
                numpy.asarray(center_of_mass_x_px, dtype=numpy.float64).copy()
            )
            self._center_of_mass_y_px.append(
                numpy.asarray(center_of_mass_y_px, dtype=numpy.float64).copy()
            )


def compute_diffraction_moments(
    indexes: DiffractionIndexes,
    patterns: DiffractionPatterns,
    *,
    bad_pixels: BadPixels,
    background: DetectorBackground | None = None,
    flat_field: FlatField | None = None,
    chunk_size: int = 256,
) -> DiffractionMoments | None:
    """Accumulate the statistics of an in-memory pattern stack, in one chunked pass.

    A *background* or *flat_field* the patterns still carry is applied first, as
    ``(pattern - background) / flat_field`` clipped at zero, and the pixels those maps
    make unusable join *bad_pixels*. Rows come out in the order of *patterns*. Returns
    None for an empty stack.
    """
    corrections = DetectorCorrections(
        background=background,
        flat_field=flat_field,
        subtract_background=True,
        divide_flat_field=True,
    )
    height, width = patterns.shape[-2:]
    accumulator = DiffractionMomentAccumulator(
        height,
        width,
        bad_pixels=corrections.apply_to_bad_pixels(bad_pixels),
        chunk_size=chunk_size,
    )

    for start in range(0, patterns.shape[0], chunk_size):
        stop = min(start + chunk_size, patterns.shape[0])
        accumulator.add(indexes[start:stop], corrections.apply(patterns[start:stop]))

    return accumulator.get_moments()
