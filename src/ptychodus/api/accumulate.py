"""Online accumulation of the per-pixel and per-pattern statistics of a diffraction stream.

One pass over the patterns, in chunks of any size and in any order, gathers everything
a coarse estimate of the probe and the object needs: the running mean, minimum and
maximum patterns, and each pattern's total and intensity centroid keyed by its scan index.
Optionally each pattern is also kept binned, the bright-field images a parallax estimate
of the defocus is made from.
Every per-pattern quantity depends only on its own pattern, so nothing has to be seen
twice and partial results from separate workers merge exactly.
"""

from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
import threading
from typing import cast

import numpy

from .assemble import (
    AssembledDiffractionData,
    DetectorCorrections,
    estimate_incident_photon_count,
)
from .diffraction import (
    BadPixels,
    DetectorBackground,
    DiffractionIndexes,
    DiffractionPattern,
    DiffractionPatterns,
    FlatField,
)
from .geometry import ImageExtent
from .inpaint import BiharmonicInpainter
from .moments import PatternStatistics, compute_intensity_moments
from .preprocess.diffraction import BinningStep, inpaint_bad_pixels
from .typing import IntegerArrayType, RealArrayType

__all__ = [
    'AssemblyMomentAccumulator',
    'DiffractionMomentAccumulator',
    'DiffractionMoments',
    'IncidentBeamEstimate',
    'compute_diffraction_moments',
    'compute_i0_scale',
    'estimate_incident_beam',
    'iterate_diffraction_moments',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiffractionMoments:
    """A snapshot of accumulated diffraction statistics; frozen, and picklable.

    The minimum, mean and maximum patterns are over the patterns as accumulated --
    corrected for any background and flat field, before bad pixels were filled;
    `bad_pixels` is the mask the accumulator inpainted with. Per-pattern rows pair with
    probe positions by `indexes`, never by order. Detector sums cover the whole pattern
    with its bad pixels inpainted. Centroids are in detector pixels about the DC pixel
    ``N // 2``, x along columns and y along rows; a pattern with no signal has a NaN
    centroid.

    With a `binning` step, `binned_patterns` holds every pattern binned by it, as float32
    and row-aligned with `indexes`. Binning follows :class:`BinningStep`: bad pixels
    count as zero, a bin touching any of them is flagged, and a flagged bin is NaN.
    """

    num_patterns: int
    minimum_pattern: DiffractionPattern
    mean_pattern: RealArrayType
    maximum_pattern: DiffractionPattern
    bad_pixels: BadPixels
    indexes: IntegerArrayType
    detector_sums: RealArrayType
    center_of_mass_x_px: RealArrayType
    center_of_mass_y_px: RealArrayType
    binning: BinningStep | None = None
    binned_patterns: RealArrayType | None = None

    def inpainted_mean_pattern(self) -> RealArrayType:
        """The mean pattern with its bad pixels inpainted.

        Inpainting is linear for a fixed mask, so this equals the mean of the inpainted
        patterns exactly, while costing one fill rather than one per pattern.
        """
        # A float64 pattern inpaints to float64; the overload only promises the dtype family.
        return cast(RealArrayType, inpaint_bad_pixels(self.mean_pattern, self.bad_pixels))

    def select(self, indexes: Sequence[int] | IntegerArrayType) -> DiffractionMoments:
        """Keep the per-pattern rows whose index is in *indexes*, in their stored order.

        The minimum, mean and maximum patterns stay those of every pattern accumulated:
        the mean pattern describes the probe, not any one position, so a subset of
        positions does not change it.
        """
        keep = numpy.isin(self.indexes, numpy.asarray(indexes))
        return replace(
            self,
            indexes=self.indexes[keep],
            detector_sums=self.detector_sums[keep],
            center_of_mass_x_px=self.center_of_mass_x_px[keep],
            center_of_mass_y_px=self.center_of_mass_y_px[keep],
            binned_patterns=None if self.binned_patterns is None else self.binned_patterns[keep],
        )

    def sort_by_index(self) -> DiffractionMoments:
        """Put the per-pattern rows in ascending order of their scan indexes.

        The sort is stable, so rows sharing an index keep their stored order. The minimum,
        mean and maximum patterns are unchanged.
        """
        order = numpy.argsort(self.indexes, kind='stable')
        return replace(
            self,
            indexes=self.indexes[order],
            detector_sums=self.detector_sums[order],
            center_of_mass_x_px=self.center_of_mass_x_px[order],
            center_of_mass_y_px=self.center_of_mass_y_px[order],
            binned_patterns=None if self.binned_patterns is None else self.binned_patterns[order],
        )

    def normalize_by_i0(
        self, i0_indexes: IntegerArrayType, i0_counts: RealArrayType
    ) -> DiffractionMoments:
        """Scale each total by ``mean(I0) / I0_j`` so a change in the beam does not read as absorption.

        Binned patterns, when kept, are scaled by the same factor.

        *i0_counts* are the incident-beam readings an upstream ion chamber, diode or scaler
        records, in any units, keyed by *i0_indexes*; each index must be unique. A reading
        that is not a positive finite number cannot correct anything and leaves its detector sum
        unscaled, with a warning, as does a pattern whose index has no reading.
        """
        scale = compute_i0_scale(self.indexes, i0_indexes, i0_counts)
        binned_patterns = self.binned_patterns

        if binned_patterns is not None:
            binned_patterns = (binned_patterns * scale[:, numpy.newaxis, numpy.newaxis]).astype(
                binned_patterns.dtype
            )

        return replace(
            self, detector_sums=self.detector_sums * scale, binned_patterns=binned_patterns
        )

    def estimate_incident_photon_count(self, *, quantile: float = 0.99) -> float:
        """The unattenuated incident photon count; see :func:`estimate_incident_photon_count`."""
        return estimate_incident_photon_count(self.detector_sums, quantile=quantile)


class DiffractionMomentAccumulator:
    """Thread-safe online accumulation of the statistics a :class:`DiffractionMoments` holds.

    Patterns are added in chunks of any size and from any number of threads. With
    *corrections*, each is corrected as it is reduced, and *bad_pixels* is widened by
    the pixels the corrections make unusable; without, patterns are added already
    corrected. They are reduced in chunks of *chunk_size*, buffered across additions
    and cut out of large ones, so the float64 working copies scale with the chunk
    rather than with what one call adds. The biharmonic fill of the bad pixels is
    factorized once and solved once per chunk.

    Each full chunk is reduced outside the lock, and its pattern statistics and per-pattern
    rows are folded in together under it, so a snapshot taken at any moment describes
    exactly the patterns its rows list.

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
        corrections: DetectorCorrections | None = None,
        chunk_size: int = 256,
        binning: BinningStep | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError('Chunks must hold at least one pattern!')

        if binning is not None:
            binned_extent = binning.apply_to_extent(ImageExtent(width_px=width, height_px=height))

            if binned_extent.width_px < 1 or binned_extent.height_px < 1:
                raise ValueError(f'{binning!r} leaves no whole bin in the patterns!')

        self._shape = (height, width)
        initial_bad_pixels: BadPixels = (
            numpy.zeros(self._shape, dtype=numpy.bool) if bad_pixels is None else bad_pixels
        )

        if initial_bad_pixels.shape != self._shape:
            raise ValueError(
                f'Bad pixels of shape {initial_bad_pixels.shape} do not match the '
                f'{self._shape} patterns!'
            )

        self._corrections = corrections
        self._bad_pixels: BadPixels = (
            initial_bad_pixels
            if corrections is None
            else corrections.apply_to_bad_pixels(initial_bad_pixels)
        )

        # The mask is fixed, so its biharmonic system is factorized once for every chunk.
        self._inpainter = (
            BiharmonicInpainter(self._bad_pixels) if numpy.any(self._bad_pixels) else None
        )
        self._chunk_size = chunk_size
        self._binning = binning
        self._binned_bad_pixels = None if binning is None else binning.apply(self._bad_pixels)

        self._pattern_statistics = PatternStatistics(height, width)
        self._lock = threading.Lock()
        self._pending_indexes: list[IntegerArrayType] = []
        self._pending_patterns: list[DiffractionPatterns] = []
        self._num_pending = 0
        self._indexes: list[IntegerArrayType] = []
        self._detector_sums: list[RealArrayType] = []
        self._center_of_mass_x_px: list[RealArrayType] = []
        self._center_of_mass_y_px: list[RealArrayType] = []
        self._binned_patterns: list[RealArrayType] = []

    @classmethod
    def from_assembled_data(
        cls,
        data: AssembledDiffractionData,
        *,
        chunk_size: int = 256,
        binning: BinningStep | None = None,
    ) -> DiffractionMomentAccumulator:
        """An empty accumulator for patterns assembled into *data*.

        It takes the pattern shape and bad pixels of *data* as they stand, and applies the
        background and flat field the patterns still carry, as
        :func:`iterate_diffraction_moments` does. No pattern is added.
        """
        _, height, width = data.get_patterns_shape()
        return cls(
            height,
            width,
            bad_pixels=data.get_bad_pixels(),
            corrections=_carried_corrections(data.get_background(), data.get_flat_field()),
            chunk_size=chunk_size,
            binning=binning,
        )

    def add(self, indexes: DiffractionIndexes, patterns: DiffractionPatterns) -> None:
        """Buffer patterns, keyed by their scan indexes, and reduce every chunk that fills."""
        if patterns.shape[-2:] != self._shape:
            raise ValueError(
                f'Patterns of shape {patterns.shape[-2:]} do not match the {self._shape} patterns!'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Expected one index per pattern!')

        if patterns.shape[0] == 0:
            return

        with self._lock:
            self._pending_indexes.append(numpy.asarray(indexes))
            self._pending_patterns.append(patterns)
            self._num_pending += patterns.shape[0]
            chunks = []

            while self._num_pending >= self._chunk_size:
                chunks.append(self._take_pending(self._chunk_size))

        for chunk in chunks:
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

        self._fold(
            PatternStatistics.from_patterns(
                moments.num_patterns,
                moments.minimum_pattern,
                moments.mean_pattern,
                moments.maximum_pattern,
            ),
            moments.indexes,
            moments.detector_sums,
            moments.center_of_mass_x_px,
            moments.center_of_mass_y_px,
            moments.binned_patterns,
        )

    def get_moments(self) -> DiffractionMoments | None:
        """Reduce any partial chunk and snapshot the statistics, or None if nothing was added.

        Accumulation may continue afterwards; the snapshot does not change with it.
        """
        with self._lock:
            chunk = self._take_pending(self._num_pending) if self._num_pending > 0 else None

        if chunk is not None:
            self._reduce(*chunk)

        # The statistics and the row lists are read together, so they describe the same
        # patterns; the rows are never mutated once folded, so they are joined outside.
        with self._lock:
            statistics = self._pattern_statistics.get_statistics()
            num_patterns = self._pattern_statistics.num_patterns
            indexes = list(self._indexes)
            detector_sums = list(self._detector_sums)
            center_of_mass_x_px = list(self._center_of_mass_x_px)
            center_of_mass_y_px = list(self._center_of_mass_y_px)
            binned_patterns = list(self._binned_patterns)

        if statistics is None:
            return None

        minimum_pattern, mean_pattern, maximum_pattern = statistics
        return DiffractionMoments(
            num_patterns=num_patterns,
            minimum_pattern=minimum_pattern,
            mean_pattern=mean_pattern,
            maximum_pattern=maximum_pattern,
            bad_pixels=self._bad_pixels.copy(),
            indexes=numpy.concatenate(indexes),
            detector_sums=numpy.concatenate(detector_sums),
            center_of_mass_x_px=numpy.concatenate(center_of_mass_x_px),
            center_of_mass_y_px=numpy.concatenate(center_of_mass_y_px),
            binning=self._binning,
            binned_patterns=None if self._binning is None else numpy.concatenate(binned_patterns),
        )

    def _take_pending(self, count: int) -> tuple[IntegerArrayType, DiffractionPatterns]:
        # Called with the lock held. Takes the first `count` pending patterns as views,
        # copying only to join pieces from separate additions.
        index_pieces: list[IntegerArrayType] = []
        pattern_pieces: list[DiffractionPatterns] = []
        remaining = count

        while remaining > 0:
            indexes = self._pending_indexes[0]
            patterns = self._pending_patterns[0]
            num_taken = min(remaining, indexes.shape[0])
            index_pieces.append(indexes[:num_taken])
            pattern_pieces.append(patterns[:num_taken])

            if num_taken == indexes.shape[0]:
                del self._pending_indexes[0]
                del self._pending_patterns[0]
            else:
                self._pending_indexes[0] = indexes[num_taken:]
                self._pending_patterns[0] = patterns[num_taken:]

            remaining -= num_taken

        self._num_pending -= count

        if len(pattern_pieces) == 1:
            return index_pieces[0], pattern_pieces[0]

        return numpy.concatenate(index_pieces), numpy.concatenate(pattern_pieces)

    def _reduce(self, indexes: IntegerArrayType, patterns: DiffractionPatterns) -> None:
        if self._corrections is not None:
            patterns = self._corrections.apply(patterns)

        height, width = self._shape
        chunk_statistics = PatternStatistics(height, width)
        chunk_statistics.add(patterns)
        weights = patterns.astype(numpy.float64, order='C')
        binned_patterns: RealArrayType | None = None

        if self._binning is not None and self._binned_bad_pixels is not None:
            zeroed = numpy.where(self._bad_pixels, 0.0, weights)
            binned_patterns = numpy.where(
                self._binned_bad_pixels, numpy.nan, self._binning.apply(zeroed)
            )

        # Binned first: the fill below overwrites the weights in place.
        repaired = (
            weights if self._inpainter is None else self._inpainter(weights, overwrite_input=True)
        )
        moments = compute_intensity_moments(repaired)

        self._fold(
            chunk_statistics,
            indexes,
            moments.total,
            moments.centroid_x_px,
            moments.centroid_y_px,
            binned_patterns,
        )

    def _fold(
        self,
        pattern_statistics: PatternStatistics,
        indexes: IntegerArrayType,
        detector_sums: RealArrayType,
        center_of_mass_x_px: RealArrayType,
        center_of_mass_y_px: RealArrayType,
        binned_patterns: RealArrayType | None,
    ) -> None:
        with self._lock:
            self._pattern_statistics.merge(pattern_statistics)
            self._indexes.append(numpy.asarray(indexes).copy())
            self._detector_sums.append(numpy.asarray(detector_sums, dtype=numpy.float64).copy())
            self._center_of_mass_x_px.append(
                numpy.asarray(center_of_mass_x_px, dtype=numpy.float64).copy()
            )
            self._center_of_mass_y_px.append(
                numpy.asarray(center_of_mass_y_px, dtype=numpy.float64).copy()
            )

            if binned_patterns is not None:
                self._binned_patterns.append(numpy.asarray(binned_patterns, dtype=numpy.float32))


class AssemblyMomentAccumulator:
    """Accumulates diffraction moments while a dataset assembles, so the pass overlaps the reads.

    :meth:`add_assembled` has the signature of the ``on_array_assembled`` callback of
    :func:`~ptychodus.api.assemble.assemble_dataset`, and may be passed as it or called
    from one; it runs on the assembly workers, as each array lands. The first array
    fixes the bad pixels and carried maps of the accumulation.
    :meth:`get_moments` then gives the moments of the assembled data, taking the
    accumulated ones wherever they are exactly what a pass over that data would give.

    A failure to accumulate is logged and never raised, because the array did
    assemble; the moments then come from one pass at the end instead.

    Patterns still buffered toward a chunk when the last array lands are reduced in
    :meth:`get_moments`, after the reads. The default *chunk_size* keeps that remainder
    small, which costs nothing, because the factorized fill makes a chunk of a few dozen
    patterns as cheap per pattern as a chunk of hundreds.
    """

    def __init__(self, *, chunk_size: int = 32) -> None:
        self._chunk_size = chunk_size
        self._lock = threading.Lock()
        self._accumulator: DiffractionMomentAccumulator | None = None
        # The first array assembled, whose mask and maps the accumulation was set up with.
        self._first_view: AssembledDiffractionData | None = None
        self._is_usable = True

    def add_assembled(self, array_index: int, label: str, view: AssembledDiffractionData) -> None:
        """Fold one assembled array into the moments; thread-safe."""
        try:
            with self._lock:
                if not self._is_usable:
                    return

                if self._accumulator is None or self._first_view is None:
                    accumulator = DiffractionMomentAccumulator.from_assembled_data(
                        view, chunk_size=self._chunk_size
                    )
                    self._accumulator = accumulator
                    self._first_view = view
                elif _same_corrections(view, self._first_view):
                    accumulator = self._accumulator
                else:
                    # This array masked pixels the others did not -- values found non-finite
                    # -- so no one mask describes what was accumulated.
                    self._is_usable = False
                    return

            accumulator.add(view.get_indexes(), view.get_patterns())
        except Exception:
            logger.exception(
                f'Failed to accumulate the diffraction moments of "{label}"; '
                'they will be computed once assembly finishes.'
            )
            self._is_usable = False

    def get_moments(self, data: AssembledDiffractionData) -> DiffractionMoments | None:
        """The moments of the patterns *data* holds, rows sorted by scan index.

        *data* is what assembly produced, or a selection of its patterns, as a filter
        applied afterwards leaves. The accumulated moments, narrowed to the indexes of
        *data*, serve when they were taken under its bad pixels and carried maps and
        cover exactly its patterns; their minimum, mean and maximum patterns are then
        those of every pattern assembled. Otherwise, as when a pixel found non-finite
        widened the mask after arrays were accumulated, the moments come from one pass
        over *data*, as :func:`compute_diffraction_moments` takes them. Returns None when
        *data* holds no patterns.
        """
        if data.get_num_patterns() == 0:
            return None

        with self._lock:
            accumulator = (
                self._accumulator
                if self._is_usable
                and self._first_view is not None
                and _same_corrections(data, self._first_view)
                else None
            )

        indexes = data.get_indexes()

        if accumulator is not None:
            try:
                moments = accumulator.get_moments()
            except Exception:
                logger.exception('Failed to reduce the accumulated diffraction moments!')
                moments = None

            if moments is not None:
                moments = moments.select(indexes)

                if moments.indexes.size == indexes.size:
                    logger.info('Using the diffraction moments accumulated during assembly.')
                    return moments.sort_by_index()

        logger.info('Accumulating diffraction moments...')
        moments = compute_diffraction_moments(
            indexes,
            data.get_patterns(),
            bad_pixels=data.get_bad_pixels(),
            background=data.get_background(),
            flat_field=data.get_flat_field(),
            chunk_size=self._chunk_size,
        )
        return None if moments is None else moments.sort_by_index()


def _same_map(map_a: RealArrayType | None, map_b: RealArrayType | None) -> bool:
    """Whether two optional per-pixel maps are both absent or equal."""
    if map_a is None or map_b is None:
        return map_a is None and map_b is None

    return numpy.array_equal(map_a, map_b)


def _same_corrections(data_a: AssembledDiffractionData, data_b: AssembledDiffractionData) -> bool:
    """Whether two assembled datasets share their bad pixels and carried maps."""
    return (
        numpy.array_equal(data_a.get_bad_pixels(), data_b.get_bad_pixels())
        and _same_map(data_a.get_background(), data_b.get_background())
        and _same_map(data_a.get_flat_field(), data_b.get_flat_field())
    )


def _carried_corrections(
    background: DetectorBackground | None, flat_field: FlatField | None
) -> DetectorCorrections:
    """Corrections applying a background and flat field the patterns still carry."""
    return DetectorCorrections(
        background=background,
        flat_field=flat_field,
        subtract_background=True,
        divide_flat_field=True,
    )


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

    corrections = _carried_corrections(background, flat_field)
    num_patterns = patterns.shape[0]
    height, width = patterns.shape[-2:]
    accumulator = DiffractionMomentAccumulator(
        height,
        width,
        bad_pixels=bad_pixels,
        corrections=corrections,
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
            accumulator.add(indexes[chunk_start:chunk_stop], patterns[chunk_start:chunk_stop])

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
    opacity carries. `moments` are those given, with the detector sums corrected for beam decay
    by I0 when `decay_corrected` says I0 was recorded.
    """

    moments: DiffractionMoments
    incident_photon_count: float
    mean_amplitude_transmission: float
    decay_corrected: bool


def estimate_incident_beam(
    moments: DiffractionMoments,
    i0_by_index: Mapping[int, float] | None = None,
    *,
    i0_photons_per_count: float | None = None,
    quantile: float = 0.99,
    max_intensity_transmission: float = 1.05,
    min_intensity_transmission: float = 1.0e-3,
    roundoff_floor: float = 1.0e-12,
) -> IncidentBeamEstimate:
    """Estimate the incident photon count and the sample's mean transmission from the detector sums.

    *i0_by_index* holds the I0 readings keyed by scan index. Where they are given
    the detector sums are first corrected for beam decay, scaled by ``mean(I0) / I0_j``. Then
    one of three rules applies, decided by the inputs alone and named in the log:

    - **I0 with a calibration.** A positive *i0_photons_per_count* converts I0 to
      incident photons: the incident photon count is ``factor * mean(I0)`` and the
      intensity transmission ``mean(total_j / (factor * I0_j))``. A transmission
      outside [*min_intensity_transmission*, *max_intensity_transmission*] -- detector sums
      above the incident beam, or a vanishing fraction of it -- is reported but kept,
      because the calibration is a statement about the instrument.
    - **I0 without a calibration.** The brightest corrected detector sums stand in for the
      empty beam: the incident photon count is their *quantile*.
    - **No I0.** The same, over the uncorrected detector sums; a calibration is unused.

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

    if i0_by_index:
        i0_indexes = numpy.fromiter(i0_by_index.keys(), dtype=numpy.intp)
        i0_counts = numpy.fromiter(i0_by_index.values(), dtype=numpy.float64)
        corrected = moments.normalize_by_i0(i0_indexes, i0_counts)
        order = numpy.argsort(i0_indexes)
        i0 = _pair_by_index(moments.indexes, i0_indexes[order], i0_counts[order])

    usable = numpy.isfinite(i0) & (i0 > roundoff_floor)
    decay_corrected = bool(usable.any())
    is_calibrated = i0_photons_per_count is not None and i0_photons_per_count > 0.0

    if decay_corrected and is_calibrated and i0_photons_per_count is not None:
        incident_photons = i0_photons_per_count * i0[usable]
        incident_photon_count = float(numpy.mean(incident_photons))
        intensity_transmission = numpy.mean(moments.detector_sums[usable] / incident_photons)
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
        detector_sums = corrected.detector_sums
        incident_photon_count = estimate_incident_photon_count(detector_sums, quantile=quantile)
        finite = detector_sums[numpy.isfinite(detector_sums)]
        intensity_transmission = (
            numpy.mean(finite) / incident_photon_count
            if finite.size > 0 and incident_photon_count > roundoff_floor
            else 1.0
        )
        source = (
            'decay-corrected detector sums'
            if decay_corrected
            else 'detector_sums; no I0 was recorded'
        )
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


def compute_i0_scale(
    indexes: IntegerArrayType, i0_indexes: IntegerArrayType, i0_counts: RealArrayType
) -> RealArrayType:
    """The factor ``mean(I0) / I0_j`` that corrects each pattern of *indexes* for beam decay.

    *i0_counts* are the incident-beam readings an upstream ion chamber, diode or scaler
    records, in any units, keyed by *i0_indexes*; each index must be unique. A reading
    that is not a positive finite number cannot correct anything and leaves its pattern
    unscaled, with a warning, as does a pattern whose index has no reading. The mean is
    over the usable readings paired with *indexes*.

    Raises:
        ValueError: if *i0_indexes* repeats an index.
    """
    unique_indexes, first = numpy.unique(i0_indexes, return_index=True)

    if unique_indexes.size != i0_indexes.size:
        raise ValueError('I0 indexes must be unique!')

    i0 = _pair_by_index(
        indexes, unique_indexes, numpy.asarray(i0_counts, dtype=numpy.float64)[first]
    )
    usable = numpy.isfinite(i0) & (i0 > 0.0)
    scale = numpy.ones(indexes.shape)

    if usable.any():
        scale[usable] = numpy.mean(i0[usable]) / i0[usable]
        logger.info('Correcting for beam decay by I0.')

    if not usable.all():
        logger.warning(
            f'{int(numpy.count_nonzero(~usable))} I0 readings are missing or not positive;'
            ' leaving those patterns uncorrected.'
        )

    return scale


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
