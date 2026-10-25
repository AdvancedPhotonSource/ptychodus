"""Online accumulation of the per-pixel and per-pattern statistics of a diffraction stream.

One pass over the patterns, in chunks of any size and in any order, gathers everything
a coarse estimate of the probe and the object needs: the running mean, minimum and
maximum frames, and each pattern's total and intensity centroid keyed by its scan index.
Optionally each pattern is also kept binned, the bright-field images a parallax estimate
of the defocus is made from.
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

    With a `bin_factor`, `binned_patterns` holds every pattern summed over square bins
    of that many pixels a side, row-aligned with `indexes`; rows and columns past the
    last whole bin are left out. A bin's sum covers its good pixels and is divided by
    their fraction, so a partly bad bin is not dimmer than its neighbours, and a bin
    with no good pixel is NaN.
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
    bin_factor: int | None = None
    binned_patterns: RealArrayType | None = None

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
            binned_patterns=None if self.binned_patterns is None else self.binned_patterns[keep],
        )

    def normalize_by_monitor(
        self, monitor_indexes: IntegerArrayType, monitor_counts: RealArrayType
    ) -> DiffractionMoments:
        """Scale each total by ``mean(M) / M_j`` so a change in the beam does not read as absorption.

        Binned patterns, when kept, are scaled by the same factor.

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
        scale = numpy.ones(self.indexes.shape)

        if usable.any():
            scale[usable] = numpy.mean(monitor[usable]) / monitor[usable]
            logger.info('Normalizing totals by the measured incident photon counts.')

        if not usable.all():
            logger.warning(
                f'{int(numpy.count_nonzero(~usable))} incident photon counts are missing or'
                ' not positive; leaving those totals unnormalized.'
            )

        binned_patterns = self.binned_patterns

        if binned_patterns is not None:
            binned_patterns = (binned_patterns * scale[:, numpy.newaxis, numpy.newaxis]).astype(
                binned_patterns.dtype
            )

        return replace(
            self, total_counts=self.total_counts * scale, binned_patterns=binned_patterns
        )

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

    A *bin_factor* also keeps every pattern binned, as float32, which costs
    ``4 / bin_factor**2`` bytes per detector pixel per pattern; see
    :class:`DiffractionMoments`.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        bad_pixels: BadPixels | None = None,
        chunk_size: int = 256,
        bin_factor: int | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError('Chunks must hold at least one pattern!')

        if bin_factor is not None and not 1 <= bin_factor <= min(height, width):
            raise ValueError(f'A bin factor of {bin_factor} does not fit the frames!')

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
        self._bin_factor = bin_factor
        self._bin_good_fraction: RealArrayType | None = None

        if bin_factor is not None:
            good = numpy.logical_not(self._bad_pixels).astype(numpy.float64)
            self._bin_good_fraction = _bin_frames(good, bin_factor) / bin_factor**2

        self._frame_statistics = FrameStatistics(height, width)
        self._lock = threading.Lock()
        self._pending_indexes: list[IntegerArrayType] = []
        self._pending_patterns: list[DiffractionPatterns] = []
        self._num_pending = 0
        self._indexes: list[IntegerArrayType] = []
        self._total_counts: list[RealArrayType] = []
        self._center_of_mass_x_px: list[RealArrayType] = []
        self._center_of_mass_y_px: list[RealArrayType] = []
        self._binned_patterns: list[RealArrayType] = []

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

        if moments.bin_factor != self._bin_factor:
            raise ValueError('Only accumulations with the same bin factor can be merged!')

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
            moments.binned_patterns,
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
                bin_factor=self._bin_factor,
                binned_patterns=(
                    None if self._bin_factor is None else numpy.concatenate(self._binned_patterns)
                ),
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
        binned_patterns: RealArrayType | None = None

        if self._bin_factor is not None and self._bin_good_fraction is not None:
            good = numpy.logical_not(self._bad_pixels)
            binned_sums = _bin_frames(numpy.where(good, patterns, 0.0), self._bin_factor)

            with numpy.errstate(divide='ignore', invalid='ignore'):
                binned = binned_sums / self._bin_good_fraction

            binned_patterns = numpy.where(self._bin_good_fraction > 0.0, binned, numpy.nan)

        self._fold_rows(
            indexes,
            moments.total,
            moments.centroid_x_px,
            moments.centroid_y_px,
            binned_patterns,
        )

    def _fold_rows(
        self,
        indexes: IntegerArrayType,
        total_counts: RealArrayType,
        center_of_mass_x_px: RealArrayType,
        center_of_mass_y_px: RealArrayType,
        binned_patterns: RealArrayType | None,
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

            if binned_patterns is not None:
                self._binned_patterns.append(numpy.asarray(binned_patterns, dtype=numpy.float32))


def _bin_frames(frames: RealArrayType, bin_factor: int) -> RealArrayType:
    """Sum the last two axes over square bins, leaving out any partial bin at the far edges."""
    height, width = frames.shape[-2:]
    num_rows = height // bin_factor
    num_columns = width // bin_factor
    trimmed = frames[..., : num_rows * bin_factor, : num_columns * bin_factor]
    blocks = trimmed.reshape(*frames.shape[:-2], num_rows, bin_factor, num_columns, bin_factor)
    return numpy.sum(blocks, axis=(-3, -1), dtype=numpy.float64)


def compute_diffraction_moments(
    indexes: DiffractionIndexes,
    patterns: DiffractionPatterns,
    *,
    bad_pixels: BadPixels,
    background: DetectorBackground | None = None,
    flat_field: FlatField | None = None,
    chunk_size: int = 256,
    bin_factor: int | None = None,
) -> DiffractionMoments | None:
    """Accumulate the statistics of an in-memory pattern stack, in one chunked pass.

    A *background* or *flat_field* the patterns still carry is applied first, as
    ``(pattern - background) / flat_field`` clipped at zero, and the pixels those maps
    make unusable join *bad_pixels*. Rows come out in the order of *patterns*. A
    *bin_factor* keeps the binned patterns as well; see :class:`DiffractionMoments`.
    Returns None for an empty stack.
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
        bin_factor=bin_factor,
    )

    for start in range(0, patterns.shape[0], chunk_size):
        stop = min(start + chunk_size, patterns.shape[0])
        accumulator.add(indexes[start:stop], corrections.apply(patterns[start:stop]))

    return accumulator.get_moments()
