"""Assemble a DiffractionDataset into an indexed pattern buffer."""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate
import concurrent.futures
import logging
import threading

import numpy

from .constants import format_bytes
from .diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionDataset,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPattern,
    DiffractionPatternCounts,
    DiffractionPatternDType,
    DiffractionPatterns,
    SimpleDiffractionArray,
)
from .geometry import ImageExtent, PixelGeometry
from .preprocess.diffraction import (
    DiffractionPrepPipeline,
    inpaint_bad_pixels,
    zero_bad_pixels,
)
from .probe_positions import ProbePositionSequence

logger = logging.getLogger(__name__)

__all__ = [
    'AssembledDiffractionData',
    'DiffractionSummary',
    'allocate_assembled_data',
    'assemble_dataset',
    'compute_array_offsets',
    'compute_assembled_patterns_shape',
    'compute_probe_photon_counts_by_index',
    'preprocess_array',
    'summarize_dataset',
]


def compute_total_counts(
    patterns: DiffractionPatterns, bad_pixels: BadPixels
) -> DiffractionPatternCounts:
    """Sum each pattern over the good (non-bad) pixels."""
    good_pixels = numpy.logical_not(bad_pixels)
    return numpy.sum(patterns[:, good_pixels], axis=-1)


def compute_probe_photon_counts_by_index(
    assembled_data: AssembledDiffractionData,
    positions: ProbePositionSequence,
) -> dict[int, float]:
    """Merge diffraction-side and position-side per-pattern counts, keyed by scan index.

    Precedence: measured diffraction-side counts, then position-side counts, then
    per-pattern total counts as fallback.
    """
    indexes = assembled_data.get_indexes()
    position_counts = positions.get_probe_photon_counts()

    if assembled_data.has_measured_probe_photon_counts():
        counts = assembled_data.get_probe_photon_counts()
        if position_counts is not None:
            logger.debug(
                'Diffraction side supplied per-pattern probe photon counts; '
                'position-side counts overridden.'
            )
        return {int(idx): float(c) for idx, c in zip(indexes, counts)}

    position_counts_by_index: dict[int, float] = {}
    if position_counts is not None:
        for position, count in zip(positions, position_counts):
            position_counts_by_index[int(position.index)] = float(count)

    total_counts = assembled_data.get_total_counts()
    return {
        int(idx): position_counts_by_index.get(int(idx), float(fallback))
        for idx, fallback in zip(indexes, total_counts)
    }


class AssembledDiffractionData:
    """In-memory store for a complete set of indexed diffraction patterns and their bad-pixel mask."""

    def __init__(
        self,
        indexes: DiffractionIndexes,
        patterns: DiffractionPatterns,
        pixel_geometry: PixelGeometry,
        bad_pixels: BadPixels,
        probe_photon_counts: DiffractionPatternCounts | None = None,
    ) -> None:
        self._indexes = indexes
        self._patterns = patterns
        self._pixel_geometry = pixel_geometry
        self._bad_pixels = bad_pixels
        self._probe_photon_counts = probe_photon_counts

        if indexes.ndim != 1:
            raise ValueError(
                f'Unexpected number of dimensions for indexes! (actual={indexes.ndim} expected=1)'
            )

        if patterns.ndim != 3:
            raise ValueError(
                f'Unexpected number of dimensions for patterns! (actual={patterns.ndim} expected=3)'
            )

        if bad_pixels.ndim != 2:
            raise ValueError(
                f'Unexpected number of dimensions for bad pixels! (actual={bad_pixels.ndim} expected=2)'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Number of indexes does not match number of patterns!')

        if patterns.shape[1:] != bad_pixels.shape:
            raise ValueError(
                'Patterns shape does not match bad pixels shape! '
                f'(actual={patterns.shape[1:]} expected={bad_pixels.shape})'
            )

        if probe_photon_counts is not None:
            if probe_photon_counts.ndim != 1:
                raise ValueError(
                    'Unexpected number of dimensions for probe photon counts! '
                    f'(actual={probe_photon_counts.ndim} expected=1)'
                )
            if probe_photon_counts.shape[0] != patterns.shape[0]:
                raise ValueError('Number of probe photon counts does not match number of patterns!')
            bad = ~numpy.isfinite(probe_photon_counts) | (probe_photon_counts < 0.0)
            if bad.any():
                first = int(numpy.argmax(bad))
                raise ValueError(
                    'Probe photon counts must be finite and non-negative; '
                    f'got {probe_photon_counts[first]!r} at position {first}.'
                )

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=numpy.intp),
            patterns=numpy.zeros((1, 1, 1), dtype=numpy.intp),
            pixel_geometry=PixelGeometry(0, 0),
            bad_pixels=numpy.zeros((1, 1), dtype=numpy.bool_),
        )

    def get_patterns_shape(self) -> tuple[int, int, int]:
        return self._patterns.shape

    def get_image_extent(self) -> ImageExtent:
        """Post-preprocessing per-pattern extent as an :class:`ImageExtent`."""
        height_px, width_px = self._patterns.shape[-2:]
        return ImageExtent(width_px=width_px, height_px=height_px)

    def get_patterns_dtype(self) -> DiffractionPatternDType:
        return self._patterns.dtype

    def get_pattern(self, index: int) -> DiffractionPattern:
        return self._patterns[index]

    def get_pixel_geometry(self) -> PixelGeometry:
        """Processed pixel geometry of the stored patterns."""
        return self._pixel_geometry

    def get_bad_pixels(self) -> BadPixels:
        return self._bad_pixels

    def assemble(self, data: AssembledDiffractionData, offset: int) -> AssembledDiffractionData:
        """Scatter a dense block into this buffer at `offset` and return a read-only view."""
        assembled_indexes = slice(offset, offset + len(data._indexes))

        self._indexes[assembled_indexes] = data._indexes
        indexes_view = self._indexes[assembled_indexes]
        indexes_view.flags.writeable = False

        self._patterns[assembled_indexes, :, :] = data._patterns
        patterns_view = self._patterns[assembled_indexes, :, :]
        patterns_view.flags.writeable = False

        probe_photon_counts_view: DiffractionPatternCounts | None = None
        if self._probe_photon_counts is not None and data._probe_photon_counts is not None:
            self._probe_photon_counts[assembled_indexes] = data._probe_photon_counts
            probe_photon_counts_view = self._probe_photon_counts[assembled_indexes]
            probe_photon_counts_view.flags.writeable = False

        return AssembledDiffractionData(
            indexes=indexes_view,
            patterns=patterns_view,
            pixel_geometry=self._pixel_geometry,
            bad_pixels=data._bad_pixels,
            probe_photon_counts=probe_photon_counts_view,
        )

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes[self._indexes >= 0]

    def get_patterns(self) -> DiffractionPatterns:
        return self._patterns[self._indexes >= 0]

    def get_total_counts(self) -> DiffractionPatternCounts:
        return compute_total_counts(self.get_patterns(), self._bad_pixels)

    def has_measured_probe_photon_counts(self) -> bool:
        """True when real hardware flux measurements are attached to this buffer."""
        return self._probe_photon_counts is not None

    def get_probe_photon_counts(self) -> DiffractionPatternCounts:
        """Per-pattern incident probe photon counts.

        Returns measured counts when :meth:`has_measured_probe_photon_counts` is
        true; otherwise falls back to the sum over good pixels for each pattern.
        """
        if self._probe_photon_counts is not None:
            return self._probe_photon_counts[self._indexes >= 0]
        return self.get_total_counts()

    def get_probe_photon_count(self) -> int:
        """Estimate the per-snapshot probe photon count.

        Returns ``max(get_probe_photon_counts())`` — measured maximum when hardware
        flux data is available, otherwise the brightest pattern's total counts over
        good pixels (which bounds the photons reaching the detector when the probe
        is least obstructed by the sample). Returns 0 when nothing is assembled.
        """
        counts = self.get_probe_photon_counts()
        return int(counts.max()) if counts.size > 0 else 0

    def get_mean_pattern(self) -> DiffractionPattern:
        assembled_patterns = self.get_patterns()
        return numpy.mean(assembled_patterns, axis=0)

    @property
    def nbytes(self) -> int:
        """Logical size of the arrays this holds.

        A memory-mapped patterns array reports its full logical size here even
        though it is backed by disk rather than RAM.
        """
        sz = self._indexes.nbytes + self._patterns.nbytes + self._bad_pixels.nbytes
        if self._probe_photon_counts is not None:
            sz += self._probe_photon_counts.nbytes
        return sz

    def __str__(self) -> str:
        number, height, width = self._patterns.shape
        dtype = str(self._patterns.dtype)
        return f'{number} x {height}H x {width}W {dtype} [{format_bytes(self._patterns.nbytes)}]'


def compute_array_offsets(metadata: DiffractionMetadata) -> Sequence[int]:
    """Return the write offset of each array, plus the total, as `N + 1` entries.

    Offsets come from ``num_patterns_per_array`` rather than a running cursor, so
    each array owns a fixed slice of the buffer. An array that yields fewer
    patterns than metadata reserves simply under-fills its slice.
    """
    return list(accumulate(metadata.num_patterns_per_array, initial=0))


def _resolve_bad_pixels(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None,
    bad_pixels: BadPixels | None,
) -> tuple[BadPixels, BadPixels]:
    """Return the (raw, processed) bad-pixel masks for a dataset."""
    raw_bad_pixels = dataset.get_bad_pixels() if bad_pixels is None else bad_pixels
    expected_shape = dataset.get_metadata().detector_extent.get_shape()

    if raw_bad_pixels.shape != expected_shape:
        raise ValueError(
            'Bad pixels shape does not match the detector extent! '
            f'(actual={raw_bad_pixels.shape} expected={expected_shape})'
        )

    processed_bad_pixels = (
        raw_bad_pixels if pipeline is None else pipeline.apply_to_mask(raw_bad_pixels)
    )
    return raw_bad_pixels, processed_bad_pixels


def _resolve_pixel_geometry(
    metadata: DiffractionMetadata,
    pipeline: DiffractionPrepPipeline | None,
    raw_pixel_geometry: PixelGeometry | None,
) -> PixelGeometry:
    """Fold the pipeline through the raw detector pixel geometry."""
    geometry = (
        metadata.detector_pixel_geometry if raw_pixel_geometry is None else raw_pixel_geometry
    )

    if geometry is None:
        raise ValueError(
            'Cannot determine the detector pixel geometry! Pass raw_pixel_geometry '
            'explicitly or set detector_pixel_geometry on the dataset metadata.'
        )

    return geometry if pipeline is None else pipeline.compute_output_pixel_geometry(geometry)


def compute_assembled_patterns_shape(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    bad_pixels: BadPixels | None = None,
) -> tuple[int, int, int]:
    """Return the `(num_patterns, height, width)` shape an assembled buffer needs.

    The frame dimensions come from the *processed* bad-pixel mask, so they already
    account for cropping, binning, padding, and transposition.
    """
    _, processed_bad_pixels = _resolve_bad_pixels(dataset, pipeline, bad_pixels)
    num_patterns_total = sum(dataset.get_metadata().num_patterns_per_array)
    height, width = processed_bad_pixels.shape
    return num_patterns_total, height, width


def allocate_assembled_data(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    bad_pixels: BadPixels | None = None,
    raw_pixel_geometry: PixelGeometry | None = None,
    patterns: DiffractionPatterns | None = None,
    dtype: DiffractionPatternDType | None = None,
) -> AssembledDiffractionData:
    """Allocate an empty buffer sized for the whole dataset.

    Pass `patterns` to supply the storage yourself -- typically a writable
    :class:`numpy.memmap` -- in which case it is used in place and never
    re-zeroed, since zeroing a fresh memmap dirties every page of its backing
    file for no benefit.

    `raw_pixel_geometry` is the *raw* detector geometry; the value stored on the
    result is the processed one. `dtype` defaults to
    ``pipeline.compute_output_dtype(metadata.pattern_dtype)``.
    """
    metadata = dataset.get_metadata()
    _, processed_bad_pixels = _resolve_bad_pixels(dataset, pipeline, bad_pixels)
    pixel_geometry = _resolve_pixel_geometry(metadata, pipeline, raw_pixel_geometry)

    num_patterns_total = sum(metadata.num_patterns_per_array)
    height, width = processed_bad_pixels.shape
    patterns_shape = (num_patterns_total, height, width)
    if dtype is not None:
        patterns_dtype = dtype
    elif pipeline is not None:
        patterns_dtype = pipeline.compute_output_dtype(metadata.pattern_dtype)
    else:
        patterns_dtype = metadata.pattern_dtype

    if patterns is None:
        logger.debug(f'Allocating {patterns_shape} {patterns_dtype} patterns buffer')
        patterns = numpy.zeros(patterns_shape, dtype=patterns_dtype)
        logger.debug(f'{format_bytes(patterns.nbytes)} allocated for patterns')
    else:
        if patterns.ndim != 3:
            raise ValueError(
                'Unexpected number of dimensions for the supplied patterns buffer! '
                f'(actual={patterns.ndim} expected=3)'
            )

        if patterns.shape != patterns_shape:
            raise ValueError(
                'Supplied patterns buffer has the wrong shape! '
                f'(actual={patterns.shape} expected={patterns_shape})'
            )

        if not patterns.flags.writeable:
            raise ValueError('Supplied patterns buffer is read-only!')

    indexes = -numpy.ones(num_patterns_total, dtype=int)

    # All-or-nothing: reserve the counts buffer only when every array reports flux.
    probe_photon_counts: DiffractionPatternCounts | None = None
    if len(dataset) > 0 and all(array.get_probe_photon_flux_Hz() is not None for array in dataset):
        probe_photon_counts = numpy.zeros(num_patterns_total, dtype=numpy.float64)

    return AssembledDiffractionData(
        indexes,
        patterns,
        pixel_geometry,
        processed_bad_pixels,
        probe_photon_counts=probe_photon_counts,
    )


def preprocess_array(
    array: DiffractionArray,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    raw_bad_pixels: BadPixels,
    processed_bad_pixels: BadPixels,
    raw_pixel_geometry: PixelGeometry,
    total_counts_lower_bound: int | None = None,
    total_counts_upper_bound: int | None = None,
    exposure_time_s: float | None = None,
) -> AssembledDiffractionData:
    """Read one array and preprocess it into a dense block of patterns.

    Bad pixels are zeroed in raw detector coordinates *before* the pipeline runs,
    then patterns outside the (inclusive) total-counts bounds are dropped.

    When ``array`` provides per-pattern probe photon flux (Hz) and
    ``exposure_time_s`` is a positive number, the flux is converted to per-pattern
    photon counts and attached to the returned block. A reader that reports flux
    without a positive exposure logs a warning and the measurement is dropped for
    that array.
    """
    label = array.get_label()
    raw_patterns = array.get_patterns()

    # Zero bad pixels in raw detector coords so saturated values don't leak into
    # neighboring bins under crop/bin/pad/flip/transpose.
    repaired_patterns = zero_bad_pixels(raw_patterns, raw_bad_pixels)
    loaded_array = SimpleDiffractionArray(label, array.get_indexes(), repaired_patterns)
    processed_array = loaded_array if pipeline is None else pipeline(loaded_array)
    indexes = processed_array.get_indexes()
    patterns = processed_array.get_patterns()

    probe_photon_counts: DiffractionPatternCounts | None = None
    probe_photon_flux_Hz = array.get_probe_photon_flux_Hz()  # noqa: N806
    if probe_photon_flux_Hz is not None:
        if exposure_time_s is not None and exposure_time_s > 0.0:
            probe_photon_counts = probe_photon_flux_Hz * exposure_time_s
        else:
            logger.warning(
                f"Dropping probe photon flux measurements from '{label}': "
                f'exposure_time_s is {exposure_time_s!r}.'
            )

    # Filter after the pipeline so counts reflect the same pattern the reconstructor sees;
    # filter every per-pattern array in lockstep to preserve the index/pattern 1:1 invariant.
    lower = total_counts_lower_bound
    upper = total_counts_upper_bound

    if lower is not None or upper is not None:
        counts = compute_total_counts(patterns, processed_bad_pixels)
        keep = numpy.ones(len(counts), dtype=bool)

        if lower is not None:
            keep &= counts >= lower

        if upper is not None:
            keep &= counts <= upper

        n_dropped = int((~keep).sum())

        if n_dropped:
            logger.info(
                f'Total counts filter dropped {n_dropped}/{len(counts)} patterns '
                f"from '{label}' (kept {int(keep.sum())})."
            )
            indexes = indexes[keep]
            patterns = patterns[keep]
            if probe_photon_counts is not None:
                probe_photon_counts = probe_photon_counts[keep]

    pixel_geometry = (
        raw_pixel_geometry
        if pipeline is None
        else pipeline.compute_output_pixel_geometry(raw_pixel_geometry)
    )
    return AssembledDiffractionData(
        indexes=indexes,
        patterns=patterns,
        pixel_geometry=pixel_geometry,
        bad_pixels=processed_bad_pixels,
        probe_photon_counts=probe_photon_counts,
    )


def _map_arrays(
    dataset: DiffractionDataset,
    work: Callable[[int, DiffractionArray], None],
    *,
    on_progress: Callable[[int, int], None] | None = None,
    on_array_error: Callable[[int, str, Exception], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_workers: int | None = None,
) -> None:
    """Run `work` over every array of a dataset on a thread pool.

    A missing array is logged and skipped. Any other per-array exception is
    reported through `on_array_error` and does not abort the remaining arrays.
    """
    num_arrays = len(dataset)

    def run_work(array_index: int, array: DiffractionArray) -> None:
        try:
            work(array_index, array)
        except FileNotFoundError:
            logger.warning(f'File not found for "{array.get_label()}"!')

    if on_progress is not None:
        on_progress(0, num_arrays)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_list = [
            executor.submit(run_work, array_index, array)
            for array_index, array in enumerate(dataset)
        ]
        future_labels = {
            future: (array_index, array.get_label())
            for future, (array_index, array) in zip(future_list, enumerate(dataset))
        }

        completed = 0
        cancelled = False

        for future in concurrent.futures.as_completed(future_list):
            array_index, label = future_labels[future]

            try:
                future.result()
            except concurrent.futures.CancelledError:
                # CancelledError derives from BaseException, so it needs its own
                # clause ahead of the generic one below.
                pass
            except Exception as ex:
                logger.warning(ex)

                if on_array_error is not None:
                    on_array_error(array_index, label, ex)

            completed += 1

            if on_progress is not None:
                on_progress(completed, num_arrays)

            if not cancelled and should_stop is not None and should_stop():
                num_cancelled = sum(1 for f in future_list if f.cancel())
                logger.info(
                    f'Stop requested; cancelled {num_cancelled} pending arrays. '
                    f'In-flight reads will finish.'
                )
                cancelled = True


def assemble_dataset(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    bad_pixels: BadPixels | None = None,
    raw_pixel_geometry: PixelGeometry | None = None,
    out: AssembledDiffractionData | None = None,
    total_counts_lower_bound: int | None = None,
    total_counts_upper_bound: int | None = None,
    on_array_assembled: Callable[[int, str, AssembledDiffractionData], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_array_error: Callable[[int, str, Exception], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_workers: int | None = None,
) -> AssembledDiffractionData:
    """Load, preprocess, and assemble every array of a dataset on a thread pool.

    Pass `out` to assemble into a buffer you allocated yourself (see
    :func:`allocate_assembled_data`); it is returned unchanged, and its pixel
    geometry wins over `raw_pixel_geometry`. Otherwise a fresh buffer is
    allocated and returned.

    A missing array is logged and skipped. Any other per-array exception is
    reported through `on_array_error` and does not abort the remaining arrays --
    this function raises only for up-front validation failures.
    """
    metadata = dataset.get_metadata()
    num_patterns_per_array = metadata.num_patterns_per_array
    num_arrays = len(dataset)

    # Streaming callers legitimately reload with an empty array list and append
    # later, so an array count *below* the metadata count is fine; above it is not.
    if num_arrays > len(num_patterns_per_array):
        raise ValueError(
            'Dataset has more arrays than metadata accounts for! '
            f'(actual={num_arrays} expected at most {len(num_patterns_per_array)})'
        )

    raw_bad_pixels, processed_bad_pixels = _resolve_bad_pixels(dataset, pipeline, bad_pixels)

    if out is None:
        data = allocate_assembled_data(
            dataset,
            pipeline,
            bad_pixels=raw_bad_pixels,
            raw_pixel_geometry=raw_pixel_geometry,
        )
    else:
        data = out

    # preprocess_array needs the raw geometry; the value on each returned block
    # is overwritten when scattered.
    block_pixel_geometry = _resolve_pixel_geometry(metadata, None, raw_pixel_geometry)
    offsets = compute_array_offsets(metadata)

    def load_array(array_index: int, array: DiffractionArray) -> None:
        label = array.get_label()
        block = preprocess_array(
            array,
            pipeline,
            raw_bad_pixels=raw_bad_pixels,
            processed_bad_pixels=processed_bad_pixels,
            raw_pixel_geometry=block_pixel_geometry,
            total_counts_lower_bound=total_counts_lower_bound,
            total_counts_upper_bound=total_counts_upper_bound,
            exposure_time_s=metadata.exposure_time_s,
        )
        num_patterns = block.get_patterns_shape()[0]
        capacity = num_patterns_per_array[array_index]

        if num_patterns > capacity:
            raise ValueError(
                f'Array "{label}" yielded more patterns than metadata reserves for it! '
                f'(actual={num_patterns} expected at most {capacity})'
            )

        view = data.assemble(block, offsets[array_index])

        if on_array_assembled is not None:
            on_array_assembled(array_index, label, view)

    _map_arrays(
        dataset,
        load_array,
        on_progress=on_progress,
        on_array_error=on_array_error,
        should_stop=should_stop,
        max_workers=max_workers,
    )

    return data


class _FrameStatistics:
    """Thread-safe running minimum/mean/maximum over chunks of patterns.

    Each chunk is reduced outside the lock; the lock is held only long enough to
    fold that chunk's three frames into the running totals.
    """

    def __init__(self, height: int, width: int) -> None:
        self._lock = threading.Lock()
        self._sum_frame = numpy.zeros((height, width), dtype=numpy.float64)
        self._minimum_frame: DiffractionPattern | None = None
        self._maximum_frame: DiffractionPattern | None = None
        self._num_patterns = 0

    def add(self, patterns: DiffractionPatterns) -> None:
        """Fold a non-empty chunk of patterns into the running statistics."""
        chunk_minimum = patterns.min(axis=0)
        chunk_maximum = patterns.max(axis=0)
        chunk_sum = patterns.sum(axis=0, dtype=numpy.float64)
        num_patterns = patterns.shape[0]

        with self._lock:
            self._sum_frame += chunk_sum
            self._num_patterns += num_patterns

            if self._minimum_frame is None or self._maximum_frame is None:
                self._minimum_frame = chunk_minimum
                self._maximum_frame = chunk_maximum
            else:
                self._minimum_frame = numpy.minimum(self._minimum_frame, chunk_minimum)
                self._maximum_frame = numpy.maximum(self._maximum_frame, chunk_maximum)

    def get_frames(
        self,
    ) -> tuple[DiffractionPattern, DiffractionPattern, DiffractionPattern] | None:
        """Return the `(minimum, mean, maximum)` frames, or None if nothing was added.

        The mean divides by the number of patterns actually folded in, not by the
        number metadata reserves, so skipped arrays do not bias it toward zero.
        """
        if self._minimum_frame is None or self._maximum_frame is None:
            return None

        return self._minimum_frame, self._sum_frame / self._num_patterns, self._maximum_frame


@dataclass(frozen=True)
class DiffractionSummary:
    """Per-pixel and per-pattern statistics for a diffraction dataset.

    The frame-shaped arrays have bad-pixel positions filled by
    :func:`~ptychodus.api.preprocess.diffraction.inpaint_bad_pixels` so the maps stay smooth for display and
    thresholding. Per-pattern ``total_counts`` sum only the good pixels.
    """

    minimum_pattern: DiffractionPattern
    mean_pattern: DiffractionPattern
    maximum_pattern: DiffractionPattern
    indexes: DiffractionIndexes
    total_counts: DiffractionPatternCounts


def summarize_dataset(
    dataset: DiffractionDataset,
    *,
    bad_pixels: BadPixels | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_array_error: Callable[[int, str, Exception], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_workers: int | None = None,
) -> DiffractionSummary:
    """Summarize a diffraction dataset with per-pixel and per-pattern statistics.

    Streams the arrays on a thread pool so the full dataset never has to be held
    in memory. Bad-pixel positions in the frame-shaped min/mean/max arrays are
    inpainted with biharmonic fill; per-pattern totals are summed over the good
    pixels only.

    Per-pattern results are written at the offsets metadata reserves and then
    compacted, so they come back in array order whatever the completion order,
    with the slots of skipped and under-filling arrays elided.
    """
    metadata = dataset.get_metadata()
    num_patterns_per_array = metadata.num_patterns_per_array
    num_arrays = len(dataset)

    if num_arrays > len(num_patterns_per_array):
        raise ValueError(
            'Dataset has more arrays than metadata accounts for! '
            f'(actual={num_arrays} expected at most {len(num_patterns_per_array)})'
        )

    # Passing no pipeline makes the raw and processed masks identical; this is
    # here for the detector-extent validation and the `bad_pixels` override.
    raw_bad_pixels, _ = _resolve_bad_pixels(dataset, None, bad_pixels)
    height, width = raw_bad_pixels.shape

    num_patterns_total = sum(num_patterns_per_array)
    indexes = -numpy.ones(num_patterns_total, dtype=numpy.intp)
    total_counts = numpy.zeros(num_patterns_total, dtype=numpy.float64)

    statistics = _FrameStatistics(height, width)
    offsets = compute_array_offsets(metadata)

    def summarize_array(array_index: int, array: DiffractionArray) -> None:
        patterns = array.get_patterns()
        num_patterns = patterns.shape[0]

        if num_patterns == 0:
            return

        capacity = num_patterns_per_array[array_index]

        if num_patterns > capacity:
            raise ValueError(
                f'Array "{array.get_label()}" yielded more patterns than metadata '
                f'reserves for it! (actual={num_patterns} expected at most {capacity})'
            )

        # Reduce before publishing anything, so a read that fails partway leaves
        # this array's slots untouched rather than half-filled.
        array_indexes = array.get_indexes()
        array_total_counts = compute_total_counts(patterns, raw_bad_pixels)

        offset = offsets[array_index]
        stop = offset + num_patterns
        indexes[offset:stop] = array_indexes
        total_counts[offset:stop] = array_total_counts
        statistics.add(patterns)

    _map_arrays(
        dataset,
        summarize_array,
        on_progress=on_progress,
        on_array_error=on_array_error,
        should_stop=should_stop,
        max_workers=max_workers,
    )

    summarized = indexes >= 0
    frames = statistics.get_frames()

    if frames is None:
        return DiffractionSummary(
            minimum_pattern=numpy.zeros((height, width), dtype=numpy.float64),
            mean_pattern=numpy.zeros((height, width), dtype=numpy.float64),
            maximum_pattern=numpy.zeros((height, width), dtype=numpy.float64),
            indexes=indexes[summarized],
            total_counts=total_counts[summarized],
        )

    minimum_frame, mean_frame, maximum_frame = frames

    return DiffractionSummary(
        minimum_pattern=inpaint_bad_pixels(minimum_frame, raw_bad_pixels),
        mean_pattern=inpaint_bad_pixels(mean_frame, raw_bad_pixels),
        maximum_pattern=inpaint_bad_pixels(maximum_frame, raw_bad_pixels),
        indexes=indexes[summarized],
        total_counts=total_counts[summarized],
    )
