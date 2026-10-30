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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
import threading
from typing import cast

import numpy

from .assemble import DetectorCorrections, estimate_incident_photon_count
from .diffraction import (
    BadPixels,
    DetectorBackground,
    DiffractionIndexes,
    DiffractionPattern,
    DiffractionPatterns,
    FlatField,
)
from .geometry import ImageExtent
from .moments import FrameStatistics, compute_intensity_moments
from .preprocess.diffraction import BinningStep, inpaint_bad_pixels
from .typing import IntegerArrayType, RealArrayType

__all__ = [
    'DiffractionMomentAccumulator',
    'DiffractionMoments',
    'IncidentBeamEstimate',
    'compute_diffraction_moments',
    'estimate_incident_beam',
    'iterate_diffraction_moments',
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

    With a `binning` step, `binned_patterns` holds every pattern binned by it, as float32
    and row-aligned with `indexes`. Binning follows :class:`BinningStep`: bad pixels
    count as zero, a bin touching any of them is flagged, and a flagged bin is NaN.
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
    binning: BinningStep | None = None
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

    def normalize_by_i0(
        self, i0_indexes: IntegerArrayType, i0_counts: RealArrayType
    ) -> DiffractionMoments:
        """Scale each total by ``mean(I0) / I0_j`` so a change in the beam does not read as absorption.

        Binned patterns, when kept, are scaled by the same factor.

        *i0_counts* are the incident-beam readings an upstream ion chamber, diode or scaler
        records, in any units, keyed by *i0_indexes*; each index must be unique. A reading
        that is not a positive finite number cannot correct anything and leaves its total
        unscaled, with a warning, as does a pattern whose index has no reading.
        """
        unique_indexes, first = numpy.unique(i0_indexes, return_index=True)

        if unique_indexes.size != i0_indexes.size:
            raise ValueError('I0 indexes must be unique!')

        i0 = _pair_by_index(
            self.indexes, unique_indexes, numpy.asarray(i0_counts, dtype=numpy.float64)[first]
        )
        usable = numpy.isfinite(i0) & (i0 > 0.0)
        scale = numpy.ones(self.indexes.shape)

        if usable.any():
            scale[usable] = numpy.mean(i0[usable]) / i0[usable]
            logger.info('Correcting the totals for beam decay by I0.')

        if not usable.all():
            logger.warning(
                f'{int(numpy.count_nonzero(~usable))} I0 readings are missing or not positive;'
                ' leaving those totals uncorrected.'
            )

        binned_patterns = self.binned_patterns

        if binned_patterns is not None:
            binned_patterns = (binned_patterns * scale[:, numpy.newaxis, numpy.newaxis]).astype(
                binned_patterns.dtype
            )

        return replace(
            self, total_counts=self.total_counts * scale, binned_patterns=binned_patterns
        )

    def estimate_incident_photon_count(self, *, quantile: float = 0.99) -> float:
        """The unattenuated incident photon count; see :func:`estimate_incident_photon_count`."""
        return estimate_incident_photon_count(self.total_counts, quantile=quantile)


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

    A *binning* step also keeps every pattern binned, as float32, which costs four
    bytes per bin per pattern; see :class:`DiffractionMoments`.
    """

    def __init__(
        self,
        height: int,
        width: int,
        *,
        bad_pixels: BadPixels | None = None,
        chunk_size: int = 256,
        binning: BinningStep | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError('Chunks must hold at least one pattern!')

        if binning is not None:
            binned_extent = binning.apply_to_extent(ImageExtent(width_px=width, height_px=height))

            if binned_extent.width_px < 1 or binned_extent.height_px < 1:
                raise ValueError(f'{binning!r} leaves no whole bin in the frames!')

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
        self._binning = binning
        self._binned_bad_pixels = None if binning is None else binning.apply(self._bad_pixels)

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

        if moments.binning != self._binning:
            raise ValueError('Only accumulations with the same binning can be merged!')

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
                binning=self._binning,
                binned_patterns=(
                    None if self._binning is None else numpy.concatenate(self._binned_patterns)
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

        if self._binning is not None and self._binned_bad_pixels is not None:
            zeroed = numpy.where(self._bad_pixels, 0.0, patterns.astype(numpy.float64))
            binned_patterns = numpy.where(
                self._binned_bad_pixels, numpy.nan, self._binning.apply(zeroed)
            )

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


def iterate_diffraction_moments(
    indexes: DiffractionIndexes,
    patterns: DiffractionPatterns,
    *,
    bad_pixels: BadPixels,
    background: DetectorBackground | None = None,
    flat_field: FlatField | None = None,
    chunk_size: int = 256,
    binning: BinningStep | None = None,
    num_snapshots: int = 1,
) -> Iterator[tuple[int, DiffractionMoments]]:
    """Accumulate an in-memory pattern stack in one chunked pass, snapshotting as it goes.

    Yields ``(num_seen, moments)`` after each of *num_snapshots* evenly spaced stops, the
    last after every pattern, so a caller can act on a partial accumulation while the
    rest is still to come; the final snapshot equals what
    :func:`compute_diffraction_moments` returns. A *background* or *flat_field* the
    patterns still carry is applied first, as ``(pattern - background) / flat_field``
    clipped at zero, and the pixels those maps make unusable join *bad_pixels*. Rows come
    out in the order of *patterns*. A *binning* step keeps the binned patterns as well;
    see :class:`DiffractionMoments`. Yields nothing for an empty stack.

    Raises:
        ValueError: if *num_snapshots* is less than one.
    """
    if num_snapshots < 1:
        raise ValueError('At least one snapshot must be taken!')

    corrections = DetectorCorrections(
        background=background,
        flat_field=flat_field,
        subtract_background=True,
        divide_flat_field=True,
    )
    num_patterns = patterns.shape[0]
    height, width = patterns.shape[-2:]
    accumulator = DiffractionMomentAccumulator(
        height,
        width,
        bad_pixels=corrections.apply_to_bad_pixels(bad_pixels),
        chunk_size=chunk_size,
        binning=binning,
    )
    stops = numpy.unique(
        numpy.rint(numpy.linspace(0, num_patterns, num_snapshots + 1)[1:]).astype(int)
    )
    start = 0

    for stop in stops[stops > 0]:
        for chunk_start in range(start, stop, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, stop)
            accumulator.add(
                indexes[chunk_start:chunk_stop],
                corrections.apply(patterns[chunk_start:chunk_stop]),
            )

        start = int(stop)
        moments = accumulator.get_moments()

        if moments is not None:
            yield start, moments


def compute_diffraction_moments(
    indexes: DiffractionIndexes,
    patterns: DiffractionPatterns,
    *,
    bad_pixels: BadPixels,
    background: DetectorBackground | None = None,
    flat_field: FlatField | None = None,
    chunk_size: int = 256,
    binning: BinningStep | None = None,
) -> DiffractionMoments | None:
    """Accumulate the statistics of an in-memory pattern stack, in one chunked pass.

    The single snapshot of :func:`iterate_diffraction_moments`, with the same
    corrections, ordering and binning. Returns None for an empty stack.
    """
    moments: DiffractionMoments | None = None

    for _, moments in iterate_diffraction_moments(
        indexes,
        patterns,
        bad_pixels=bad_pixels,
        background=background,
        flat_field=flat_field,
        chunk_size=chunk_size,
        binning=binning,
    ):
        pass

    return moments


@dataclass(frozen=True)
class IncidentBeamEstimate:
    """The incident beam a diffraction scan implies, and the moments it was judged from.

    `incident_photon_count` is the photons per exposure the probe is scaled to, and
    `mean_amplitude_transmission` the modulus a uniform object of the sample's mean
    opacity carries. `moments` are those given, with the totals corrected for beam decay
    by I0 when `decay_corrected` says I0 was recorded.
    """

    moments: DiffractionMoments
    incident_photon_count: float
    mean_amplitude_transmission: float
    decay_corrected: bool


def estimate_incident_beam(
    moments: DiffractionMoments,
    i0_counts_by_index: Mapping[int, float] | None = None,
    *,
    i0_photons_per_count: float | None = None,
    quantile: float = 0.99,
    max_intensity_transmission: float = 1.05,
    min_intensity_transmission: float = 1.0e-3,
    roundoff_floor: float = 1.0e-12,
) -> IncidentBeamEstimate:
    """Estimate the incident photon count and the sample's mean transmission from the totals.

    *i0_counts_by_index* holds the I0 readings keyed by scan index. Where they are given
    the totals are first corrected for beam decay, scaled by ``mean(I0) / I0_j``. Then
    one of three rules applies, decided by the inputs alone and named in the log:

    - **I0 with a calibration.** A positive *i0_photons_per_count* converts I0 to
      incident photons: the incident photon count is ``factor * mean(I0)`` and the
      intensity transmission ``mean(total_j / (factor * I0_j))``. A transmission
      outside [*min_intensity_transmission*, *max_intensity_transmission*] -- totals
      above the incident beam, or a vanishing fraction of it -- is reported but kept,
      because the calibration is a statement about the instrument.
    - **I0 without a calibration.** The brightest corrected totals stand in for the
      empty beam: the incident photon count is their *quantile*.
    - **No I0.** The same, over the uncorrected totals; a calibration is unused.

    Without a calibration the intensity transmission is the mean total over the
    incident photon count. Its square root, clipped to [0, 1], is the mean amplitude
    transmission. An incident photon count at or below *roundoff_floor* gives a
    transparent sample rather than a division by it.

    Raises:
        ValueError: if *quantile* is outside ``[0, 1]`` or the transmission bounds are
            out of order.
    """
    if min_intensity_transmission > max_intensity_transmission:
        raise ValueError('The minimum intensity transmission exceeds the maximum!')

    corrected = moments
    i0 = numpy.full(moments.indexes.shape, numpy.nan)

    if i0_counts_by_index:
        i0_indexes = numpy.fromiter(i0_counts_by_index.keys(), dtype=numpy.intp)
        i0_counts = numpy.fromiter(i0_counts_by_index.values(), dtype=numpy.float64)
        corrected = moments.normalize_by_i0(i0_indexes, i0_counts)
        order = numpy.argsort(i0_indexes)
        i0 = _pair_by_index(moments.indexes, i0_indexes[order], i0_counts[order])

    usable = numpy.isfinite(i0) & (i0 > roundoff_floor)
    decay_corrected = bool(usable.any())
    is_calibrated = i0_photons_per_count is not None and i0_photons_per_count > 0.0

    if decay_corrected and is_calibrated and i0_photons_per_count is not None:
        incident_photons = i0_photons_per_count * i0[usable]
        incident_photon_count = float(numpy.mean(incident_photons))
        intensity_transmission = numpy.mean(moments.total_counts[usable] / incident_photons)
        logger.info(
            f'Incident photon count {incident_photon_count:.4g} from I0 at '
            f'{i0_photons_per_count:.4g} photons per count.'
        )

        if not min_intensity_transmission <= intensity_transmission <= max_intensity_transmission:
            logger.warning(
                f'The calibrated I0 implies an intensity transmission of '
                f'{intensity_transmission:.4g}, outside [{min_intensity_transmission:.4g}, '
                f'{max_intensity_transmission:.4g}]; check the I0 calibration.'
            )
    else:
        totals = corrected.total_counts
        incident_photon_count = estimate_incident_photon_count(totals, quantile=quantile)
        finite = totals[numpy.isfinite(totals)]
        intensity_transmission = (
            numpy.mean(finite) / incident_photon_count
            if finite.size > 0 and incident_photon_count > roundoff_floor
            else 1.0
        )
        source = 'decay-corrected totals' if decay_corrected else 'totals; no I0 was recorded'
        logger.info(
            f'Incident photon count {incident_photon_count:.4g} from the {quantile:g} '
            f'quantile of the {source}.'
        )

    return IncidentBeamEstimate(
        moments=corrected,
        incident_photon_count=incident_photon_count,
        mean_amplitude_transmission=float(numpy.sqrt(numpy.clip(intensity_transmission, 0, 1))),
        decay_corrected=decay_corrected,
    )


def _pair_by_index(
    indexes: IntegerArrayType, sorted_indexes: IntegerArrayType, values: RealArrayType
) -> RealArrayType:
    """The value for each of *indexes* from *values* keyed by unique *sorted_indexes*; NaN if none."""
    paired = numpy.full(indexes.shape, numpy.nan)

    if sorted_indexes.size == 0:
        return paired

    position = numpy.clip(numpy.searchsorted(sorted_indexes, indexes), 0, sorted_indexes.size - 1)
    found = sorted_indexes[position] == indexes
    paired[found] = values[position[found]]
    return paired
