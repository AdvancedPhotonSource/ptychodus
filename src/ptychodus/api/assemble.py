"""Assemble a DiffractionDataset into an indexed pattern buffer."""

from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import cast, TypeVar
import concurrent.futures
import logging
import os
import threading

import numpy

from .constants import format_bytes
from .diffraction import (
    BadPixels,
    BeamCenter,
    CropRegion,
    DetectorBackground,
    DiffractionArray,
    DiffractionDataset,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPattern,
    DiffractionPatternCounts,
    DiffractionPatternDType,
    DiffractionPatterns,
    FlatField,
    SimpleDiffractionArray,
    validate_pixel_map,
)
from .geometry import ImageExtent, PixelGeometry
from .moments import FrameStatistics
from .preprocess.diffraction import (
    DiffractionPrepPipeline,
    FilterValuesStep,
    SignalRadius,
    estimate_signal_radius,
    inpaint_bad_pixels,
    zero_bad_pixels,
)
from .typing import RealArrayType

logger = logging.getLogger(__name__)

__all__ = [
    'AssembledDiffractionData',
    'CropSuggestion',
    'DetectorCorrections',
    'DiffractionSummary',
    'DiffractionTotalCounts',
    'allocate_assembled_data',
    'assemble_dataset',
    'compute_array_offsets',
    'compute_assembled_dtype',
    'compute_assembled_patterns_shape',
    'compute_dataset_total_counts',
    'default_max_workers',
    'estimate_incident_photon_count',
    'preprocess_array',
    'suggest_crop_region',
    'summarize_dataset',
]


_ArrayT = TypeVar('_ArrayT', bound=numpy.ndarray)


def _read_only(array: _ArrayT) -> _ArrayT:
    """Return `array` as non-writeable, through a view rather than in place.

    Downgrading the view leaves the base writable, so taking a read-only handle on a
    buffer does not stop the next scatter from filling it.
    """
    view = array[...]
    view.flags.writeable = False
    return cast(_ArrayT, view)


def compute_total_counts(
    patterns: DiffractionPatterns, bad_pixels: BadPixels
) -> DiffractionPatternCounts:
    """Sum each pattern over the good (non-bad) pixels."""
    if not bad_pixels.any() and numpy.issubdtype(patterns.dtype, numpy.integer):
        # Nothing is masked, so this reduces over every pixel and agrees with the masked
        # form below to the bit -- without the (num_patterns, num_good) copy that boolean
        # indexing materializes whatever the mask holds. Integers only: the two summation
        # orders do not agree to the last bit in floating point, and these counts are
        # compared against a threshold.
        return numpy.sum(patterns, axis=(-2, -1))

    good_pixels = numpy.logical_not(bad_pixels)
    return numpy.sum(patterns[:, good_pixels], axis=-1)


def estimate_incident_photon_count(total_counts: RealArrayType, *, quantile: float = 0.99) -> float:
    """Estimate the unattenuated incident photon count from per-pattern totals.

    A pattern recorded through empty beam -- or, failing that, through the most
    transparent part of the sample -- is among the brightest, so a high quantile of the
    totals stands in for the incident flux. A quantile just below one keeps a single
    saturated, cosmic-ray or otherwise anomalous frame from setting it; ``1.0`` takes
    the brightest frame itself. NaN totals are ignored, and no finite total gives 0.

    Raises:
        ValueError: if *quantile* is outside ``[0, 1]``.
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f'quantile must lie in [0, 1], got {quantile}')

    finite = total_counts[numpy.isfinite(total_counts)]

    if finite.size == 0:
        return 0.0

    return float(numpy.quantile(finite, quantile))


class AssembledDiffractionData:
    """In-memory store for a complete set of indexed diffraction patterns and their bad-pixel mask."""

    def __init__(
        self,
        indexes: DiffractionIndexes,
        patterns: DiffractionPatterns,
        pixel_geometry: PixelGeometry,
        bad_pixels: BadPixels,
        *,
        background: DetectorBackground | None = None,
        flat_field: FlatField | None = None,
    ) -> None:
        self._indexes = indexes
        self._patterns = patterns
        self._pixel_geometry = pixel_geometry
        self._bad_pixels = bad_pixels
        self._background = background
        self._flat_field = flat_field

        if indexes.ndim != 1:
            raise ValueError(
                f'Unexpected number of dimensions for indexes! (actual={indexes.ndim} expected=1)'
            )

        if patterns.ndim != 3:
            raise ValueError(
                f'Unexpected number of dimensions for patterns! (actual={patterns.ndim} expected=3)'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Number of indexes does not match number of patterns!')

        validate_pixel_map(bad_pixels, patterns.shape[1:], name='Bad pixels')

        validate_pixel_map(background, bad_pixels.shape, name='Background')
        validate_pixel_map(flat_field, bad_pixels.shape, name='Flat field')

        if background is not None and not numpy.isfinite(background).all():
            raise ValueError('Background must be finite.')

        if flat_field is not None and not numpy.isfinite(flat_field).all():
            raise ValueError('Flat field must be finite.')

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=numpy.intp),
            patterns=numpy.zeros((1, 1, 1), dtype=numpy.intp),
            pixel_geometry=PixelGeometry(0, 0),
            bad_pixels=numpy.zeros((1, 1), dtype=numpy.bool),
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

    def get_background(self) -> DetectorBackground | None:
        """Processed detector background not yet subtracted from the patterns, if any."""
        return None if self._background is None else _read_only(self._background)

    def get_flat_field(self) -> FlatField | None:
        """Processed flat-field response the patterns have not been divided by, if any."""
        return None if self._flat_field is None else _read_only(self._flat_field)

    def add_bad_pixels(self, bad_pixels: BadPixels) -> None:
        """Mark more pixels bad, in addition to those already masked.

        The mask is replaced rather than written in place, since the array held here
        may be shared with whoever supplied it.
        """
        validate_pixel_map(bad_pixels, self._bad_pixels.shape, name='Bad pixels')
        self._bad_pixels = numpy.logical_or(self._bad_pixels, bad_pixels)

    def assemble(self, data: AssembledDiffractionData, offset: int) -> AssembledDiffractionData:
        """Scatter a dense block into this buffer at `offset` and return a read-only view."""
        assembled_indexes = slice(offset, offset + len(data._indexes))

        self._indexes[assembled_indexes] = data._indexes
        indexes_view = _read_only(self._indexes[assembled_indexes])

        self._patterns[assembled_indexes, :, :] = data._patterns
        patterns_view = _read_only(self._patterns[assembled_indexes, :, :])

        return AssembledDiffractionData(
            indexes=indexes_view,
            patterns=patterns_view,
            pixel_geometry=self._pixel_geometry,
            bad_pixels=data._bad_pixels,
            background=self._background,
            flat_field=self._flat_field,
        )

    def _every_slot_assembled(self) -> bool:
        """Whether every slot holds a pattern, so no slot needs eliding.

        Recomputed rather than cached because :meth:`assemble` writes the index array in
        place, from a thread pool, and the same slice can be scattered more than once --
        no cached flag would see either. It costs one pass over the indexes, one number
        per pattern, against the whole-buffer copy it decides whether to make.
        """
        return bool((self._indexes >= 0).all())

    def get_num_patterns(self) -> int:
        """How many patterns are assembled, counted without materializing them.

        This is what :meth:`get_patterns` returns the length of. The first element of
        :meth:`get_patterns_shape` is the buffer's capacity instead, which is larger
        whenever a slot went unfilled.
        """
        return int(numpy.count_nonzero(self._indexes >= 0))

    def get_indexes(self) -> DiffractionIndexes:
        """Scan indexes of the assembled patterns, in buffer order. Non-writeable.

        See :meth:`get_patterns` for why.
        """
        if self._every_slot_assembled():
            return _read_only(self._indexes)

        return _read_only(self._indexes[self._indexes >= 0])

    def get_patterns(self) -> DiffractionPatterns:
        """The assembled patterns, in buffer order. Non-writeable.

        Eliding unassembled slots means boolean indexing, which copies the whole buffer;
        with nothing to elide the buffer itself is the answer, so this returns a view over
        live storage whenever it can. Writing through that view would rewrite what every
        other holder sees -- and, for the memory-mapped buffers
        :func:`allocate_assembled_data` accepts, what is on disk.

        The gathered copy is marked the same way, so what a caller may do with the result
        does not depend on how full the buffer happened to be.
        """
        if self._every_slot_assembled():
            return _read_only(self._patterns)

        return _read_only(self._patterns[self._indexes >= 0])

    def get_total_counts(self) -> DiffractionPatternCounts:
        return compute_total_counts(self.get_patterns(), self._bad_pixels)

    def get_mean_pattern(self) -> DiffractionPattern:
        """Mean over the assembled patterns, or a zero frame when none are assembled.

        The empty case is reachable whenever the total-counts filter drops every
        pattern of an array; `numpy.mean` would return an all-NaN frame there and
        poison every downstream display and weighted average.
        """
        assembled_patterns = self.get_patterns()

        if assembled_patterns.shape[0] == 0:
            return numpy.zeros(assembled_patterns.shape[-2:], dtype=numpy.float64)

        return numpy.mean(assembled_patterns, axis=0)

    @property
    def nbytes(self) -> int:
        """Logical size of the arrays this holds.

        A memory-mapped patterns array reports its full logical size here even
        though it is backed by disk rather than RAM.
        """
        sz = self._indexes.nbytes + self._patterns.nbytes + self._bad_pixels.nbytes
        if self._background is not None:
            sz += self._background.nbytes
        if self._flat_field is not None:
            sz += self._flat_field.nbytes
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


@dataclass(frozen=True)
class DetectorCorrections:
    """A measured detector background and flat field, and whether to apply them.

    The flat field is the detector's response to uniform illumination, so a pattern
    is corrected as ``(pattern - background) / flat_field`` and clipped at zero.
    Correction happens after the preprocessing pipeline, with both maps carried
    through it, so value filters and bad-pixel repair act on measured counts. Maps
    that are not applied travel with the assembled data instead, for consumers that
    model them.

    The maps start on the raw detector extent (see :meth:`crop` and
    :meth:`through_pipeline`). A non-finite map entry marks its pixel bad, as does a
    flat-field response at or below `flat_field_floor` when dividing, because
    dividing by it would amplify noise without bound; values at those pixels are
    never used.
    """

    background: DetectorBackground | None = None
    flat_field: FlatField | None = None
    subtract_background: bool = False
    divide_flat_field: bool = False
    flat_field_floor: float = 1.0e-3

    @classmethod
    def create_null(cls) -> DetectorCorrections:
        """No maps, so nothing is applied or carried."""
        return cls()

    @property
    def applies_background(self) -> bool:
        return self.subtract_background and self.background is not None

    @property
    def applies_flat_field(self) -> bool:
        return self.divide_flat_field and self.flat_field is not None

    def apply_to_dtype(self, dtype: DiffractionPatternDType) -> DiffractionPatternDType:
        """Return the pattern dtype after correction: floating once anything is applied."""
        if not (self.applies_background or self.applies_flat_field):
            return dtype

        return numpy.dtype(numpy.float64 if dtype == numpy.float64 else numpy.float32)

    def apply_to_bad_pixels(self, bad_pixels: BadPixels) -> BadPixels:
        """Return `bad_pixels` widened by the pixels whose map entries are unusable."""
        widened = bad_pixels

        if self.background is not None:
            widened = numpy.logical_or(widened, numpy.logical_not(numpy.isfinite(self.background)))

        if self.flat_field is not None:
            unusable = numpy.logical_not(numpy.isfinite(self.flat_field))

            if self.divide_flat_field:
                unusable = numpy.logical_or(
                    unusable, numpy.nan_to_num(self.flat_field, nan=0.0) <= self.flat_field_floor
                )

            widened = numpy.logical_or(widened, unusable)

        return widened

    def apply(self, patterns: DiffractionPatterns) -> DiffractionPatterns:
        """Subtract and divide as configured, clipping at zero; a no-op when nothing applies.

        The maps must be on the same grid as `patterns`. Values at pixels
        :meth:`apply_to_bad_pixels` flags are meaningless, as they are for any bad pixel.
        """
        background = self.background if self.subtract_background else None
        flat_field = self.flat_field if self.divide_flat_field else None

        if background is None and flat_field is None:
            return patterns

        # Past the early return at least one map applies, so apply_to_dtype has promoted to
        # floating. The union it is declared to return cannot say that, and the in-place
        # divide below is rejected on an integer array.
        corrected = cast(RealArrayType, patterns.astype(self.apply_to_dtype(patterns.dtype)))

        if background is not None:
            neutral_background = numpy.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0)
            corrected -= neutral_background.astype(corrected.dtype)

        if flat_field is not None:
            neutral_flat_field = numpy.nan_to_num(flat_field, nan=1.0, posinf=1.0, neginf=1.0)
            divisor = numpy.where(
                neutral_flat_field > self.flat_field_floor, neutral_flat_field, 1.0
            )
            corrected /= divisor.astype(corrected.dtype)

        numpy.clip(corrected, 0, None, out=corrected)
        return corrected

    def crop(self, read_region: CropRegion | None) -> DetectorCorrections:
        """Return these corrections with both maps cut to `read_region`."""
        if read_region is None:
            return self

        return replace(
            self,
            background=None if self.background is None else read_region.apply_to(self.background),
            flat_field=None if self.flat_field is None else read_region.apply_to(self.flat_field),
        )

    def through_pipeline(self, pipeline: DiffractionPrepPipeline | None) -> DetectorCorrections:
        """Return these corrections with both maps carried onto the pipeline's output grid.

        The background is summed wherever the pipeline combines pixels and is zero
        where it pads; the flat field is averaged and is one where it pads. Non-finite
        entries become neutral values, since their pixels are bad.
        """
        background = None
        flat_field = None

        if self.background is not None:
            background = numpy.nan_to_num(self.background, nan=0.0, posinf=0.0, neginf=0.0)

            if pipeline is not None:
                background = pipeline.apply_to_pixel_map(background)

        if self.flat_field is not None:
            flat_field = numpy.nan_to_num(self.flat_field, nan=1.0, posinf=1.0, neginf=1.0)

            if pipeline is not None:
                summed = pipeline.apply_to_pixel_map(flat_field)
                weights = pipeline.apply_to_pixel_map(numpy.ones_like(flat_field))
                flat_field = numpy.divide(
                    summed, weights, out=numpy.ones_like(summed), where=weights > 0
                )

        return replace(self, background=background, flat_field=flat_field)


@dataclass(frozen=True)
class _ResolvedCorrections:
    """Corrections cut to the read region, and the same carried through the pipeline.

    `raw` widens the raw bad-pixel mask; `processed` corrects the processed patterns.
    The carried maps are the processed ones that are not applied.
    """

    raw: DetectorCorrections
    processed: DetectorCorrections

    @property
    def carried_background(self) -> DetectorBackground | None:
        return None if self.processed.applies_background else self.processed.background

    @property
    def carried_flat_field(self) -> FlatField | None:
        return None if self.processed.applies_flat_field else self.processed.flat_field


def _resolve_corrections(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None,
    corrections: DetectorCorrections | None,
    read_region: CropRegion | None = None,
) -> _ResolvedCorrections:
    """Return corrections cropped to `read_region`, and carried through `pipeline`.

    None means the dataset's own maps, carried rather than applied.
    """
    if corrections is None:
        corrections = DetectorCorrections(
            background=dataset.get_background(), flat_field=dataset.get_flat_field()
        )

    expected_shape = dataset.get_metadata().detector_extent.get_shape()

    validate_pixel_map(corrections.background, expected_shape, name='Background')
    validate_pixel_map(corrections.flat_field, expected_shape, name='Flat field')

    raw = corrections.crop(read_region)
    return _ResolvedCorrections(raw, raw.through_pipeline(pipeline))


def _resolve_bad_pixels(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None,
    bad_pixels: BadPixels | None,
    read_region: CropRegion | None = None,
    corrections: DetectorCorrections | None = None,
) -> tuple[BadPixels, BadPixels]:
    """Return the (raw-in-read-region, processed) bad-pixel masks for a dataset.

    When `read_region` is set, the raw mask is first cropped to that region so the
    returned pair matches what a reader with load-time cropping delivers: the
    "raw" mask is the cropped detector mask (still pre-pipeline), and the
    "processed" mask is that cropped mask run through the pipeline.

    `corrections`, already cropped to `read_region`, widen the raw mask by the
    pixels whose map entries are unusable before the pipeline runs.
    """
    raw_bad_pixels = dataset.get_bad_pixels() if bad_pixels is None else bad_pixels
    expected_shape = dataset.get_metadata().detector_extent.get_shape()

    validate_pixel_map(raw_bad_pixels, expected_shape, name='Bad pixels')

    if read_region is not None:
        raw_bad_pixels = read_region.apply_to(raw_bad_pixels)

    if corrections is not None:
        raw_bad_pixels = corrections.apply_to_bad_pixels(raw_bad_pixels)

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
    read_region: CropRegion | None = None,
) -> tuple[int, int, int]:
    """Return the `(num_patterns, height, width)` shape an assembled buffer needs.

    The frame dimensions come from the *processed* bad-pixel mask, so they already
    account for load-time cropping (`read_region`), binning, padding, and
    transposition. Detector corrections never change the shape.
    """
    _, processed_bad_pixels = _resolve_bad_pixels(dataset, pipeline, bad_pixels, read_region)
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
    read_region: CropRegion | None = None,
    corrections: DetectorCorrections | None = None,
) -> AssembledDiffractionData:
    """Allocate an empty buffer sized for the whole dataset.

    Pass `patterns` to supply the storage yourself -- typically a writable
    :class:`numpy.memmap` -- in which case it is used in place and never
    re-zeroed, since zeroing a fresh memmap dirties every page of its backing
    file for no benefit.

    `raw_pixel_geometry` is the *raw* detector geometry; the value stored on the
    result is the processed one. `dtype` defaults to :func:`compute_assembled_dtype`.

    `corrections` of None means the dataset's own background and flat field,
    carried rather than applied. Whichever maps are not applied are stored on the
    result in processed coordinates; the stored mask includes the pixels the maps
    make unusable.
    """
    metadata = dataset.get_metadata()
    resolved = _resolve_corrections(dataset, pipeline, corrections, read_region)
    _, processed_bad_pixels = _resolve_bad_pixels(
        dataset, pipeline, bad_pixels, read_region, resolved.raw
    )
    pixel_geometry = _resolve_pixel_geometry(metadata, pipeline, raw_pixel_geometry)

    num_patterns_total = sum(metadata.num_patterns_per_array)
    height, width = processed_bad_pixels.shape
    patterns_shape = (num_patterns_total, height, width)
    patterns_dtype = (
        compute_assembled_dtype(dataset, pipeline, corrections=resolved.raw)
        if dtype is None
        else dtype
    )

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

    return AssembledDiffractionData(
        indexes,
        patterns,
        pixel_geometry,
        processed_bad_pixels,
        background=resolved.carried_background,
        flat_field=resolved.carried_flat_field,
    )


def compute_assembled_dtype(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    corrections: DetectorCorrections | None = None,
) -> DiffractionPatternDType:
    """Return the pattern dtype an assembled buffer needs to hold every result.

    The pipeline widens the measured dtype for any step that sums; detector
    corrections, applied after it, then make the patterns floating point.
    """
    dtype = dataset.get_metadata().pattern_dtype

    if pipeline is not None:
        dtype = pipeline.compute_output_dtype(dtype)

    return dtype if corrections is None else corrections.apply_to_dtype(dtype)


def _zero_nonfinite_values(
    patterns: DiffractionPatterns,
) -> tuple[DiffractionPatterns, BadPixels | None]:
    """Zero every NaN or infinite value, and return the pixels that held one in any frame.

    Integer patterns cannot hold such values and are returned untouched, as are
    floating patterns that hold none; the second element is then None.
    """
    if not numpy.issubdtype(patterns.dtype, numpy.inexact):
        return patterns, None

    nonfinite = numpy.logical_not(numpy.isfinite(patterns))

    if not nonfinite.any():
        return patterns, None

    cleaned = patterns.copy()
    cleaned[nonfinite] = 0
    nonfinite_pixels: BadPixels = numpy.any(nonfinite, axis=0)
    return cleaned, nonfinite_pixels


def _with_nonfinite_pixels(
    processed_bad_pixels: BadPixels,
    pipeline: DiffractionPrepPipeline | None,
    nonfinite_pixels: BadPixels | None,
) -> BadPixels:
    """Return the processed mask widened by raw-coordinate non-finite pixels."""
    if nonfinite_pixels is None:
        return processed_bad_pixels

    processed_nonfinite = (
        nonfinite_pixels if pipeline is None else pipeline.apply_to_mask(nonfinite_pixels)
    )
    return numpy.logical_or(processed_bad_pixels, processed_nonfinite)


@dataclass(frozen=True)
class _PreprocessedPatterns:
    """One array after preprocessing, with the masks that apply to it.

    `bad_pixels` is the processed mask for this array: the dataset mask widened by
    any pixel that held a non-finite value here. `nonfinite_pixels` holds those
    pixels in raw (read-region) coordinates, or None when there were none.
    """

    indexes: DiffractionIndexes
    patterns: DiffractionPatterns
    bad_pixels: BadPixels
    nonfinite_pixels: BadPixels | None


def _preprocess_patterns(
    array: DiffractionArray,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    raw_bad_pixels: BadPixels,
    processed_bad_pixels: BadPixels,
    total_counts_lower_bound: int | None = None,
    total_counts_upper_bound: int | None = None,
    read_region: CropRegion | None = None,
    corrections: DetectorCorrections | None = None,
) -> _PreprocessedPatterns:
    """Read one array and run the preprocessing pipeline over it.

    The geometry-free core of :func:`preprocess_array`: everything that shapes the
    indexes and patterns, with no detector pixel geometry
    involved. Callers that need an :class:`AssembledDiffractionData` go through
    :func:`preprocess_array`; callers that keep only per-pattern reductions --
    :func:`compute_dataset_total_counts` -- use this directly, and so need no
    pixel geometry to run.

    Bad pixels are zeroed *before* the pipeline runs; when `read_region` is set,
    the reader is asked for the crop rectangle directly (HDF5 partial read) and
    `raw_bad_pixels` must already be the same-region slice of the detector mask.
    Zero-then-crop and crop-then-zero commute for per-pixel masks, so the result
    matches full-frame preprocessing followed by cropping. Patterns outside the
    (inclusive) total-counts bounds are dropped after the pipeline runs.

    A NaN or infinite value is zeroed before anything else touches it, and its
    pixel is masked for this array; the pixel is reported back so a caller can
    mask it for the whole dataset.

    `corrections`, already on the pipeline's output grid, are applied after the
    pipeline and before the total-counts filter; the masks must already include the
    pixels the corrections make unusable (see :func:`_resolve_bad_pixels`).
    """
    label = array.get_label()
    raw_patterns, nonfinite_pixels = _zero_nonfinite_values(
        array.get_patterns(read_region=read_region)
    )
    block_bad_pixels = _with_nonfinite_pixels(processed_bad_pixels, pipeline, nonfinite_pixels)

    # Zero bad pixels before the pipeline so saturated values don't leak into
    # neighboring bins under bin/pad/flip/transpose. When read_region is set,
    # raw_bad_pixels is the cropped mask matching the pre-cropped patterns.
    repaired_patterns = zero_bad_pixels(raw_patterns, raw_bad_pixels)
    loaded_array = SimpleDiffractionArray(label, array.get_indexes(), repaired_patterns)
    processed_array = loaded_array if pipeline is None else pipeline(loaded_array)
    indexes = processed_array.get_indexes()
    patterns = processed_array.get_patterns()

    if corrections is not None:
        patterns = corrections.apply(patterns)

    # Filter after the pipeline so counts reflect the same pattern the reconstructor sees;
    # filter every per-pattern array in lockstep to preserve the index/pattern 1:1 invariant.
    lower = total_counts_lower_bound
    upper = total_counts_upper_bound

    if lower is not None or upper is not None:
        counts = compute_total_counts(patterns, block_bad_pixels)
        keep = numpy.ones(len(counts), dtype=bool)

        if lower is not None:
            keep &= counts >= lower

        if upper is not None:
            keep &= counts <= upper

        n_dropped = int((~keep).sum())
        n_kept = int(keep.sum())

        if n_dropped:
            message = (
                f'Total counts filter dropped {n_dropped}/{len(counts)} patterns '
                f"from '{label}' (kept {n_kept})."
            )

            if n_kept == 0:
                # Bounds are compared against post-crop, post-pipeline counts; bounds
                # picked from raw full-detector statistics can sit an order of
                # magnitude above the real distribution and empty the array.
                logger.warning(
                    f'{message} Check TotalCountsLowerBound/TotalCountsUpperBound'
                    ' against the cropped, preprocessed counts.'
                )
            else:
                logger.info(message)
            indexes = indexes[keep]
            patterns = patterns[keep]

    return _PreprocessedPatterns(
        indexes=indexes,
        patterns=patterns,
        bad_pixels=block_bad_pixels,
        nonfinite_pixels=nonfinite_pixels,
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
    read_region: CropRegion | None = None,
    corrections: DetectorCorrections | None = None,
    on_nonfinite_pixels: Callable[[BadPixels], None] | None = None,
) -> AssembledDiffractionData:
    """Preprocess one array into a dense block of patterns with its pixel geometry.

    Wraps :func:`_preprocess_patterns` -- which documents the preprocessing
    itself -- and folds `pipeline` through `raw_pixel_geometry` so the returned
    block carries the processed geometry matching its patterns.

    The returned block's mask includes any pixel that held a NaN or infinite value
    in this array. Those pixels are also passed, in raw read-region coordinates, to
    `on_nonfinite_pixels` when there are any.
    """
    preprocessed = _preprocess_patterns(
        array,
        pipeline,
        raw_bad_pixels=raw_bad_pixels,
        processed_bad_pixels=processed_bad_pixels,
        total_counts_lower_bound=total_counts_lower_bound,
        total_counts_upper_bound=total_counts_upper_bound,
        read_region=read_region,
        corrections=corrections,
    )

    if preprocessed.nonfinite_pixels is not None and on_nonfinite_pixels is not None:
        on_nonfinite_pixels(preprocessed.nonfinite_pixels)

    pixel_geometry = (
        raw_pixel_geometry
        if pipeline is None
        else pipeline.compute_output_pixel_geometry(raw_pixel_geometry)
    )
    return AssembledDiffractionData(
        indexes=preprocessed.indexes,
        patterns=preprocessed.patterns,
        pixel_geometry=pixel_geometry,
        bad_pixels=preprocessed.bad_pixels,
    )


def default_max_workers(
    *, max_arrays_in_flight: int = 8, local_world_size: int | None = None
) -> int:
    """How many arrays to hold in flight at once when the caller names no number.

    Each worker materializes a whole array block, so the pool size sets peak memory, and
    `max_arrays_in_flight` matters more than the throughput: an unbounded pool admits as
    many simultaneous blocks as the machine has cores. Raising it trades peak memory for
    throughput on the numpy half of the work and buys nothing on the read half, because
    h5py serializes every HDF5 call behind one lock; the preprocessing that follows each
    read is what scales, and it saturates well before the core count.

    `local_world_size` divides the budget, and defaults to what the launcher reported in
    ``LOCAL_WORLD_SIZE``. A launcher that runs one process per local GPU shares the
    node's memory between them, and each would otherwise size itself as though it were
    alone. A reported value that is empty or not an integer is ignored rather than
    fatal: it means something in the environment claims a world size it cannot state,
    and sizing the pool for a node this process has to itself is the better reading.
    """
    num_cpus = os.cpu_count() or 1

    if local_world_size is None:
        try:
            local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        except (KeyError, ValueError):
            local_world_size = 1

    return max(1, min(max_arrays_in_flight, num_cpus) // max(1, local_world_size))


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

    `max_workers` of None resolves to :func:`default_max_workers`, which explains what
    the resulting pool size costs.
    """
    num_arrays = len(dataset)

    if max_workers is None:
        max_workers = default_max_workers()

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


class _NonfinitePixelUnion:
    """Thread-safe union of the raw-coordinate non-finite pixels found across arrays."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pixels: BadPixels | None = None

    def add(self, nonfinite_pixels: BadPixels) -> None:
        with self._lock:
            if self._pixels is None:
                self._pixels = nonfinite_pixels.copy()
            else:
                numpy.logical_or(self._pixels, nonfinite_pixels, out=self._pixels)

    def get(self) -> BadPixels | None:
        return self._pixels


def _embed_in_detector(
    pixels: BadPixels, detector_extent: ImageExtent, read_region: CropRegion | None
) -> BadPixels:
    """Place a read-region mask into a detector-extent mask that is False elsewhere."""
    if read_region is None:
        return pixels

    embedded = numpy.zeros(detector_extent.get_shape(), dtype=numpy.bool)
    read_region.apply_to(embedded)[...] = pixels
    return embedded


def _warn_nonfinite_pixels(nonfinite_pixels: BadPixels) -> None:
    logger.warning(
        f'Masked {int(numpy.count_nonzero(nonfinite_pixels))} detector pixel(s) holding'
        ' NaN or infinite values; those values were zeroed.'
    )


def assemble_dataset(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    bad_pixels: BadPixels | None = None,
    raw_pixel_geometry: PixelGeometry | None = None,
    out: AssembledDiffractionData | None = None,
    total_counts_lower_bound: int | None = None,
    total_counts_upper_bound: int | None = None,
    read_region: CropRegion | None = None,
    corrections: DetectorCorrections | None = None,
    on_array_assembled: Callable[[int, str, AssembledDiffractionData], None] | None = None,
    on_nonfinite_pixels: Callable[[BadPixels], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_array_error: Callable[[int, str, Exception], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_workers: int | None = None,
) -> AssembledDiffractionData:
    """Load, preprocess, and assemble every array of a dataset on a thread pool.

    Pass `out` to assemble into a buffer you allocated yourself (see
    :func:`allocate_assembled_data`), with the same `corrections`; it is returned,
    and its pixel geometry wins over `raw_pixel_geometry`. Otherwise a fresh buffer
    is allocated and returned. `corrections` of None means the dataset's own maps,
    carried rather than applied.

    A pixel holding a NaN or infinite value in any frame is masked for the whole
    dataset once every array is assembled, and its values are zeroed. The returned
    buffer's mask includes those pixels, and they are passed, in detector-extent
    coordinates, to `on_nonfinite_pixels` when there are any. Arrays assembled
    before such a pixel turned up were filtered on total counts that still
    included it.

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

    resolved = _resolve_corrections(dataset, pipeline, corrections, read_region)
    raw_bad_pixels, processed_bad_pixels = _resolve_bad_pixels(
        dataset, pipeline, bad_pixels, read_region, resolved.raw
    )

    if out is None:
        data = allocate_assembled_data(
            dataset,
            pipeline,
            bad_pixels=bad_pixels,
            raw_pixel_geometry=raw_pixel_geometry,
            read_region=read_region,
            corrections=corrections,
        )
    else:
        data = out

    # preprocess_array needs the raw geometry; the value on each returned block
    # is overwritten when scattered.
    block_pixel_geometry = _resolve_pixel_geometry(metadata, None, raw_pixel_geometry)
    offsets = compute_array_offsets(metadata)
    nonfinite_union = _NonfinitePixelUnion()

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
            read_region=read_region,
            corrections=resolved.processed,
            on_nonfinite_pixels=nonfinite_union.add,
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

    nonfinite_pixels = nonfinite_union.get()

    if nonfinite_pixels is not None:
        _warn_nonfinite_pixels(nonfinite_pixels)
        data.add_bad_pixels(
            _with_nonfinite_pixels(processed_bad_pixels, pipeline, nonfinite_pixels)
        )

        if on_nonfinite_pixels is not None:
            on_nonfinite_pixels(
                _embed_in_detector(nonfinite_pixels, metadata.detector_extent, read_region)
            )

    return data


@dataclass(frozen=True)
class DiffractionSummary:
    """Per-pixel and per-pattern statistics for a diffraction dataset.

    The frame-shaped arrays have bad-pixel positions filled by
    :func:`~ptychodus.api.preprocess.diffraction.inpaint_bad_pixels` so the maps stay smooth for display and
    thresholding. Per-pattern ``total_counts`` sum only the good pixels.

    ``minimum_pattern`` and ``maximum_pattern`` carry the pattern dtype and
    ``mean_pattern`` is float64, whether or not the dataset had bad pixels and
    whether or not it had any patterns at all.
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
    pixels only. A pixel holding a NaN or infinite value in any frame is zeroed
    and treated as bad, as :func:`assemble_dataset` treats it.

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

    statistics = FrameStatistics(height, width)
    offsets = compute_array_offsets(metadata)
    nonfinite_union = _NonfinitePixelUnion()

    def summarize_array(array_index: int, array: DiffractionArray) -> None:
        patterns, nonfinite_pixels = _zero_nonfinite_values(array.get_patterns())
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
        array_total_counts = compute_total_counts(
            patterns, _with_nonfinite_pixels(raw_bad_pixels, None, nonfinite_pixels)
        )

        offset = offsets[array_index]
        stop = offset + num_patterns
        indexes[offset:stop] = array_indexes
        total_counts[offset:stop] = array_total_counts
        statistics.add(patterns)

        if nonfinite_pixels is not None:
            nonfinite_union.add(nonfinite_pixels)

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
        # Dtypes match the populated branch below, where the extrema come from
        # reductions over the patterns and the mean from a float64 accumulator.
        return DiffractionSummary(
            minimum_pattern=numpy.zeros((height, width), dtype=metadata.pattern_dtype),
            mean_pattern=numpy.zeros((height, width), dtype=numpy.float64),
            maximum_pattern=numpy.zeros((height, width), dtype=metadata.pattern_dtype),
            indexes=indexes[summarized],
            total_counts=total_counts[summarized],
        )

    minimum_frame, mean_frame, maximum_frame = frames
    nonfinite_pixels = nonfinite_union.get()

    if nonfinite_pixels is not None:
        _warn_nonfinite_pixels(nonfinite_pixels)

    summary_bad_pixels = _with_nonfinite_pixels(raw_bad_pixels, None, nonfinite_pixels)

    return DiffractionSummary(
        minimum_pattern=inpaint_bad_pixels(minimum_frame, summary_bad_pixels),
        mean_pattern=inpaint_bad_pixels(mean_frame, summary_bad_pixels),
        maximum_pattern=inpaint_bad_pixels(maximum_frame, summary_bad_pixels),
        indexes=indexes[summarized],
        total_counts=total_counts[summarized],
    )


@dataclass(frozen=True)
class DiffractionTotalCounts:
    """Per-pattern good-pixel totals measured under a specific prep plan.

    Distinct from the ``total_counts`` field of :class:`DiffractionSummary`, which
    sums the raw full detector: these are summed over exactly the patterns the
    assembler produces, after the load-time crop and after the pipeline. They are
    therefore the numbers the total-counts filter actually compares its bounds
    against, and the ones robust statistics should be derived from.
    """

    indexes: DiffractionIndexes
    total_counts: DiffractionPatternCounts


def compute_dataset_total_counts(
    dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None = None,
    *,
    bad_pixels: BadPixels | None = None,
    read_region: CropRegion | None = None,
    corrections: DetectorCorrections | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_array_error: Callable[[int, str, Exception], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_workers: int | None = None,
) -> DiffractionTotalCounts:
    """Measure per-pattern total counts as the assembler would see them.

    Runs the same :func:`_preprocess_patterns` core that :func:`assemble_dataset`
    reaches through :func:`preprocess_array` -- same load-time crop, same pipeline,
    same processed bad-pixel mask -- but keeps only the per-pattern totals and
    never applies the total-counts filter, since the point is to measure the
    distribution the filter bounds will be chosen from. No frame-shaped
    statistics are accumulated and no assembled buffer is allocated, and when
    ``read_region`` is set a reader with partial-read support fetches only the
    crop rectangle, so this is markedly cheaper than :func:`summarize_dataset`
    over the same dataset.

    Per-pattern results are written at the offsets metadata reserves and then
    compacted, so they come back in array order whatever the completion order,
    with the slots of skipped and under-filling arrays elided.

    Unlike :func:`assemble_dataset` and :func:`allocate_assembled_data`, this pass
    needs no detector pixel geometry: it keeps only per-pattern totals and never
    builds a geometry-bearing block. So it runs against metadata that carries no
    `detector_pixel_geometry`, which most readers never set.
    """
    metadata = dataset.get_metadata()
    num_patterns_per_array = metadata.num_patterns_per_array
    num_arrays = len(dataset)

    if num_arrays > len(num_patterns_per_array):
        raise ValueError(
            'Dataset has more arrays than metadata accounts for! '
            f'(actual={num_arrays} expected at most {len(num_patterns_per_array)})'
        )

    resolved = _resolve_corrections(dataset, pipeline, corrections, read_region)
    raw_bad_pixels, processed_bad_pixels = _resolve_bad_pixels(
        dataset, pipeline, bad_pixels, read_region, resolved.raw
    )

    num_patterns_total = sum(num_patterns_per_array)
    indexes = -numpy.ones(num_patterns_total, dtype=numpy.intp)
    total_counts = numpy.zeros(num_patterns_total, dtype=numpy.float64)

    offsets = compute_array_offsets(metadata)

    def count_array(array_index: int, array: DiffractionArray) -> None:
        preprocessed = _preprocess_patterns(
            array,
            pipeline,
            raw_bad_pixels=raw_bad_pixels,
            processed_bad_pixels=processed_bad_pixels,
            read_region=read_region,
            corrections=resolved.processed,
        )
        array_indexes = preprocessed.indexes
        num_patterns = len(array_indexes)

        if num_patterns == 0:
            return

        capacity = num_patterns_per_array[array_index]

        if num_patterns > capacity:
            raise ValueError(
                f'Array "{array.get_label()}" yielded more patterns than metadata '
                f'reserves for it! (actual={num_patterns} expected at most {capacity})'
            )

        # Reduce before publishing anything, so a read that fails partway leaves
        # this array's slots untouched rather than half-filled. This is the same
        # reduction AssembledDiffractionData.get_total_counts delegates to.
        array_total_counts = compute_total_counts(preprocessed.patterns, preprocessed.bad_pixels)

        offset = offsets[array_index]
        stop = offset + num_patterns
        indexes[offset:stop] = array_indexes
        total_counts[offset:stop] = array_total_counts

    _map_arrays(
        dataset,
        count_array,
        on_progress=on_progress,
        on_array_error=on_array_error,
        should_stop=should_stop,
        max_workers=max_workers,
    )

    counted = indexes >= 0

    return DiffractionTotalCounts(
        indexes=indexes[counted],
        total_counts=total_counts[counted],
    )


@dataclass(frozen=True)
class CropSuggestion:
    """A crop sized to the signal in a pattern, and the radius estimate it was sized from."""

    signal: SignalRadius
    region: CropRegion

    @property
    def is_capped(self) -> bool:
        """True when the signal reaches past the largest power-of-two square that fits."""
        return 2 * self.signal.radius_px > self.region.width_px


def suggest_crop_region(
    mean_pattern: DiffractionPattern,
    beam_center: BeamCenter,
    detector_extent: ImageExtent,
    *,
    bad_pixels: BadPixels,
    value_filter: FilterValuesStep | None = None,
    corrections: DetectorCorrections | None = None,
) -> CropSuggestion:
    """Suggest the power-of-two square about `beam_center` that holds the signal in a pattern.

    `mean_pattern` is a raw-detector frame such as :attr:`DiffractionSummary.mean_pattern`.
    It is prepared as assembly prepares every pattern -- cut by `value_filter`, then
    corrected by `corrections` -- and `bad_pixels` is widened by the pixels the correction
    maps leave unusable, so the estimate sees what will be reconstructed. The maps must be
    on the raw detector grid. The radius comes from :func:`estimate_signal_radius` and the
    square from :meth:`CropRegion.from_smallest_pow2`.

    Raises ``ValueError`` when no signal rises above the noise floor, or when `beam_center`
    is not inside the detector.
    """
    pattern = mean_pattern if value_filter is None else value_filter.apply(mean_pattern)

    if corrections is not None:
        pattern = corrections.apply(pattern[numpy.newaxis])[0]
        bad_pixels = corrections.apply_to_bad_pixels(bad_pixels)

    signal = estimate_signal_radius(pattern, beam_center, bad_pixels=bad_pixels)

    if signal.radius_px < 1:
        raise ValueError(
            'No signal rises above the noise floor about the beam center '
            f'({beam_center.x_px}, {beam_center.y_px}).'
        )

    region = CropRegion.from_smallest_pow2(beam_center, signal.radius_px, detector_extent)

    if region.width_px < 1:
        raise ValueError(
            f'The beam center ({beam_center.x_px}, {beam_center.y_px}) is not inside the '
            f'{detector_extent.width_px}x{detector_extent.height_px} detector.'
        )

    return CropSuggestion(signal=signal, region=region)
