"""Diffraction pattern preprocessing: the step pipeline and standalone helpers.

Steps share a single `apply` interface that returns the kind of array it was
given, and infer whether the input is a 3-D pattern stack or a 2-D boolean
bad-pixel mask from `data.dtype`. The order of operations is encoded in
`DiffractionPrepPipeline.steps`; the canonical order emitted by the model-layer
factory is filter → binning → padding → hflip → vflip → transpose.
Cropping is hoisted to load time (see :class:`DiffractionPrepPlan`) so readers
that support partial reads (HDF5) only fetch the crop rectangle from disk.

The free functions at the end of the module are helpers used *alongside* the
pipeline rather than steps within it, and they deliberately stay that way.
`zero_bad_pixels` and `inpaint_bad_pixels` are interchangeable bad-pixel
repairs that run in raw detector coordinates before the pipeline, and
`estimate_beam_center` and `estimate_signal_radius` take a whole frame at
once. A step that repaired or dropped frames would also break the index/pattern 1:1 invariant that
`prepare_reconstruct_input` relies on, because
`DiffractionPrepPipeline.__call__` passes the original indexes through
unchanged.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias, TypeVar, cast, overload

import numpy
from pydantic import BaseModel, ConfigDict, Field
from scipy import ndimage
from skimage.restoration import inpaint_biharmonic

from ..diffraction import (
    BadPixels,
    BeamCenter,
    CropRegion,
    DiffractionArray,
    DiffractionPattern,
    DiffractionPatternDType,
    DiffractionPatterns,
    PixelMap,
    SimpleDiffractionArray,
)
from ..geometry import ImageExtent, PixelGeometry
from ..typing import AnyArrayType, RealArrayType
from .noise import RobustStatistics, estimate_noise_floor

# A step rearranges or combines the last two axes and leaves rank and dtype family to the
# caller, so every `apply` returns exactly what it was handed: a pattern stack, a bad-pixel
# mask, or a per-pixel map. Binding to `ndarray` rather than to a union of the three keeps
# that promise without naming them, which is what lets the default `apply_to_pixel_map`
# below delegate to `apply`.
_PrepT = TypeVar('_PrepT', bound=numpy.ndarray)


def _is_mask(data: AnyArrayType) -> bool:
    return data.dtype == numpy.bool


class DiffractionPrepStep(BaseModel):
    """Abstract base for a single preprocessing step.

    To add a new step:

    - Declare `type: Literal['<unique_tag>'] = '<unique_tag>'`. The default is required so
      callers never pass the tag and `model_dump_json` emits it automatically.
    - Add the class to `DiffractionPrepStepUnion`, or it is unreachable on deserialize.
    - Override `apply_to_extent` / `apply_to_pixel_geometry` / `apply_to_dtype` only where the
      step is not identity in that dimension; the defaults below pass the input through unchanged.
    - Branch on `_is_mask(data)` in `apply` if pattern and mask behavior differ, as
      `BinningStep` does (sum for patterns, logical-OR for masks).
    - Override `apply_to_pixel_map` where an additive per-pixel map (a detector background)
      must not follow the pattern path; a dtype cannot tell such a map from a pattern.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def apply(self, data: _PrepT) -> _PrepT:
        """Apply this step. Mask vs pattern behavior is inferred from `data.dtype`."""

    def apply_to_pixel_map(self, data: PixelMap) -> PixelMap:
        """Apply this step to a 2-D additive per-pixel map, in floating point.

        A map is a per-pixel quantity that adds up the way counts do, so it follows
        the pattern geometrically and sums wherever pixels are combined. Default:
        :meth:`apply`, which is right for steps that only rearrange pixels.
        """
        return self.apply(data)

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        """Return the extent this step would produce given `extent`. Default: identity."""
        return extent

    def apply_to_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        """Return the pixel geometry this step would produce. Default: identity."""
        return geometry

    def apply_to_dtype(self, dtype: DiffractionPatternDType) -> DiffractionPatternDType:
        """Return the dtype this step would produce given `dtype`. Default: identity."""
        return dtype


class FilterValuesStep(DiffractionPrepStep):
    """Zero pattern values outside `[lower_bound, upper_bound)`. No-op on masks."""

    type: Literal['filter_values'] = 'filter_values'
    lower_bound: int | None = None
    upper_bound: int | None = None

    def apply(self, data: _PrepT) -> _PrepT:
        if _is_mask(data):
            return data

        if self.lower_bound is None and self.upper_bound is None:
            return data

        out = data.copy()

        if self.lower_bound is not None:
            out[out < self.lower_bound] = 0

        if self.upper_bound is not None:
            out[out >= self.upper_bound] = 0

        return out

    def apply_to_pixel_map(self, data: PixelMap) -> PixelMap:
        return data


class BinningStep(DiffractionPrepStep):
    """Reduce each `bin_size_y × bin_size_x` block. Sum for patterns; logical-OR for masks.

    A binned pixel is bad when any of its sub-pixels is: bad pixels are zeroed before the
    pipeline runs, so a partly bad block sums fewer measured pixels than a whole one, and
    flagging it keeps that undercount from passing as a measurement.

    An extent the bin size does not divide leaves a partial block at the far edge, which
    is discarded: the output holds `extent // bin_size` whole blocks, which is what
    :meth:`apply_to_extent` reports and what a caller sizing a buffer from it allocates.
    """

    type: Literal['binning'] = 'binning'
    bin_size_x: int = Field(gt=0)
    bin_size_y: int = Field(gt=0)

    def apply(self, data: _PrepT) -> _PrepT:
        binned_height = data.shape[-2] // self.bin_size_y
        binned_width = data.shape[-1] // self.bin_size_x
        # Drop the trailing partial block so the reshape below is exact.
        whole_blocks = data[
            ..., : binned_height * self.bin_size_y, : binned_width * self.bin_size_x
        ]
        shape = data.shape[:-2] + (binned_height, self.bin_size_y, binned_width, self.bin_size_x)
        reshaped = whole_blocks.reshape(shape)
        if _is_mask(data):
            return cast(_PrepT, numpy.logical_or.reduce(reshaped, axis=(-3, -1), keepdims=False))
        return cast(
            _PrepT,
            numpy.sum(
                reshaped, axis=(-3, -1), keepdims=False, dtype=self.apply_to_dtype(data.dtype)
            ),
        )

    def apply_to_pixel_map(self, data: PixelMap) -> PixelMap:
        return self.apply(data.astype(numpy.float64))

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        return ImageExtent(
            width_px=extent.width_px // self.bin_size_x,
            height_px=extent.height_px // self.bin_size_y,
        )

    def apply_to_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        return PixelGeometry(
            width_m=geometry.width_m * self.bin_size_x,
            height_m=geometry.height_m * self.bin_size_y,
        )

    def apply_to_dtype(self, dtype: DiffractionPatternDType) -> DiffractionPatternDType:
        if not numpy.issubdtype(dtype, numpy.integer):
            return dtype
        factor = self.bin_size_x * self.bin_size_y
        if factor <= 1:
            return dtype
        # issubdtype does not narrow the pattern dtype union for the checker.
        needed_max = int(numpy.iinfo(cast(numpy.dtype[numpy.integer], dtype)).max) * factor
        return numpy.promote_types(dtype, numpy.min_scalar_type(needed_max))


class PaddingStep(DiffractionPrepStep):
    """Symmetrically pad the last two axes. Fill 0 for patterns; True for masks.

    Padded pixels are bad: nothing was measured there, so flagging them keeps a zero fill
    from reading as zero photons observed.
    """

    type: Literal['padding'] = 'padding'
    pad_x: int = Field(ge=0)
    pad_y: int = Field(ge=0)

    def apply(self, data: _PrepT) -> _PrepT:
        leading_pad = ((0, 0),) * (data.ndim - 2)
        pad_width = (*leading_pad, (self.pad_y, self.pad_y), (self.pad_x, self.pad_x))
        fill = True if _is_mask(data) else 0
        return cast(_PrepT, numpy.pad(data, pad_width, mode='constant', constant_values=fill))

    def apply_to_pixel_map(self, data: PixelMap) -> PixelMap:
        pad_width = ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x))
        return numpy.pad(data, pad_width, mode='constant', constant_values=0.0)

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        return ImageExtent(
            width_px=extent.width_px + 2 * self.pad_x,
            height_px=extent.height_px + 2 * self.pad_y,
        )


class HorizontalFlipStep(DiffractionPrepStep):
    """Flip the last axis."""

    type: Literal['hflip'] = 'hflip'

    def apply(self, data: _PrepT) -> _PrepT:
        return numpy.flip(data, axis=-1)


class VerticalFlipStep(DiffractionPrepStep):
    """Flip the second-to-last axis."""

    type: Literal['vflip'] = 'vflip'

    def apply(self, data: _PrepT) -> _PrepT:
        return numpy.flip(data, axis=-2)


class TransposeStep(DiffractionPrepStep):
    """Swap the last two axes."""

    type: Literal['transpose'] = 'transpose'

    def apply(self, data: _PrepT) -> _PrepT:
        axes = tuple(range(data.ndim - 2)) + (data.ndim - 1, data.ndim - 2)
        return cast(_PrepT, numpy.transpose(data, axes=axes))

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        return ImageExtent(width_px=extent.height_px, height_px=extent.width_px)

    def apply_to_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        return PixelGeometry(width_m=geometry.height_m, height_m=geometry.width_m)


DiffractionPrepStepUnion: TypeAlias = (
    FilterValuesStep
    | BinningStep
    | PaddingStep
    | HorizontalFlipStep
    | VerticalFlipStep
    | TransposeStep
)


class DiffractionPrepPipeline(BaseModel):
    """Ordered chain of preprocessing steps applied to patterns and the bad-pixel mask."""

    model_config = ConfigDict(frozen=True)

    # The `discriminator='type'` tag keeps the field-less steps distinguishable:
    # HorizontalFlipStep, VerticalFlipStep, and TransposeStep all serialize to the same empty
    # JSON object, so shape-based union resolution would silently deserialize one as another.
    steps: tuple[Annotated[DiffractionPrepStepUnion, Field(discriminator='type')], ...] = ()

    def _apply(self, data: _PrepT) -> _PrepT:
        for step in self.steps:
            data = step.apply(data)
        return data

    def apply_to_patterns(self, patterns: DiffractionPatterns) -> DiffractionPatterns:
        """Run every step over a pattern stack. A single 2-D pattern is promoted to a stack."""
        if patterns.ndim == 2:
            patterns = patterns[numpy.newaxis, ...]
        elif patterns.ndim != 3:
            raise ValueError(f'Invalid diffraction pattern dimensions! (shape={patterns.shape})')

        return self._apply(patterns)

    def apply_to_mask(self, bad_pixels: BadPixels) -> BadPixels:
        """Run every step over a 2-D boolean bad-pixel mask."""
        if bad_pixels.ndim != 2:
            raise ValueError(f'Invalid bad_pixel dimensions! (shape={bad_pixels.shape})')

        return self._apply(bad_pixels)

    def apply_to_pixel_map(self, pixel_map: PixelMap) -> PixelMap:
        """Run every step over a 2-D additive per-pixel map, returning float64.

        Pixels the pipeline combines are summed and padded pixels are zero, which is
        what a detector background needs. A map that should average instead (a flat
        field) can be divided afterwards by this method applied to an all-ones map.
        """
        if pixel_map.ndim != 2:
            raise ValueError(f'Invalid pixel map dimensions! (shape={pixel_map.shape})')

        data: PixelMap = pixel_map.astype(numpy.float64)

        for step in self.steps:
            data = step.apply_to_pixel_map(data)

        return data

    def __call__(self, array: DiffractionArray) -> DiffractionArray:
        """Return a new array with the pipeline applied, preserving label and scan indexes."""
        patterns = self.apply_to_patterns(array.get_patterns())
        return SimpleDiffractionArray(array.get_label(), array.get_indexes(), patterns)

    def compute_output_extent(self, extent: ImageExtent) -> ImageExtent:
        """Return the extent the pipeline would produce, without touching pattern data."""
        for step in self.steps:
            extent = step.apply_to_extent(extent)
        return extent

    def compute_output_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        """Return the pixel geometry the pipeline would produce, without touching pattern data."""
        for step in self.steps:
            geometry = step.apply_to_pixel_geometry(geometry)
        return geometry

    def compute_output_dtype(self, dtype: DiffractionPatternDType) -> DiffractionPatternDType:
        """Return the dtype the pipeline would produce, sized for every step's promotion."""
        for step in self.steps:
            dtype = step.apply_to_dtype(dtype)
        return dtype


@dataclass(frozen=True)
class DiffractionPrepPlan:
    """A load-time crop region paired with the in-memory preprocessing pipeline.

    The `read_region` is passed to :meth:`DiffractionArray.get_patterns` so readers
    that support partial reads (HDF5) only fetch the crop rectangle from disk. The
    `pipeline` runs on whatever the reader returns and never re-crops.
    """

    read_region: CropRegion | None
    pipeline: DiffractionPrepPipeline


def zero_bad_pixels(
    patterns: DiffractionPatterns, bad_pixels: BadPixels | None
) -> DiffractionPatterns:
    """Return a copy of `patterns` with bad pixels zeroed.

    Zero is a neutral choice (no photons measured) but does mildly bias any
    consumer toward predicting low intensity at those locations. Returns the
    input unchanged when `bad_pixels` is None or all-False to avoid a copy.
    """
    if bad_pixels is None or not numpy.any(bad_pixels):
        return patterns

    cleaned = patterns.copy()
    cleaned[:, bad_pixels] = 0
    return cleaned


@overload
def inpaint_bad_pixels(
    data: DiffractionPattern, bad_pixels: BadPixels | None
) -> DiffractionPattern:
    """Return `data` with bad-pixel positions filled by biharmonic inpainting."""
    ...


@overload
def inpaint_bad_pixels(
    data: DiffractionPatterns, bad_pixels: BadPixels | None
) -> DiffractionPatterns:
    """Return `data` with bad-pixel positions filled by biharmonic inpainting."""
    ...


def inpaint_bad_pixels(
    data: DiffractionPattern | DiffractionPatterns, bad_pixels: BadPixels | None
) -> DiffractionPattern | DiffractionPatterns:
    """Return `data` with bad-pixel positions filled by biharmonic inpainting.

    Takes a single frame or a stack of them and returns the same shape and dtype
    it was given, so for a pattern stack this is interchangeable with
    :func:`zero_bad_pixels` -- the two differ in what they put at the bad-pixel
    positions, nothing else.

    Only interior holes are inpainted. A bad region that touches the array border
    -- padding, a detector edge, a gap running off the frame -- is zeroed instead,
    because filling it would extrapolate beyond every measured pixel rather than
    interpolate between them.

    Returns the input unchanged when `bad_pixels` is None or all-False so the
    (expensive) skimage call is skipped. A stack shares one mask across all its
    frames, so the biharmonic system is solved once for the whole stack rather
    than once per frame; the result is identical either way.

    Integer values are rounded rather than truncated, which would bias every
    filled pixel low.
    """
    if bad_pixels is None or not numpy.any(bad_pixels):
        return data

    if data.ndim not in (2, 3):
        raise ValueError('Diffraction data must be 2- or 3-dimensional ndarray.')

    labels, _ = ndimage.label(bad_pixels)
    border_labels = numpy.unique(
        numpy.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    edge_pixels = numpy.isin(labels, border_labels[border_labels != 0])
    interior_pixels = numpy.logical_and(bad_pixels, numpy.logical_not(edge_pixels))

    if not numpy.any(interior_pixels):
        filled = data.astype(numpy.float64)
    elif data.ndim == 2:
        filled = inpaint_biharmonic(data.astype(numpy.float64), interior_pixels)
    else:
        stacked = inpaint_biharmonic(
            numpy.moveaxis(data, 0, -1).astype(numpy.float64),
            interior_pixels,
            channel_axis=-1,
        )
        filled = numpy.moveaxis(stacked, -1, 0)

    filled[..., edge_pixels] = 0.0

    dtype = data.dtype

    if numpy.issubdtype(dtype, numpy.integer):
        # Fills are already confined to the range of each frame's good pixels, so
        # they are representable; the clip costs one pass and keeps the cast from
        # wrapping silently if that ever stops being true.
        # issubdtype does not narrow the pattern dtype union for the checker.
        iinfo = numpy.iinfo(cast(numpy.dtype[numpy.integer], dtype))
        filled = numpy.clip(numpy.rint(filled), iinfo.min, iinfo.max)

    return filled.astype(dtype)


def estimate_beam_center(
    pattern: DiffractionPattern,
    *,
    mad_threshold: float = 4.5,
) -> BeamCenter:
    """Estimate the direct-beam pixel of a diffraction pattern.

    Pipeline: 3x3 median filter -> Otsu-based noise-floor estimate ->
    2-pass intensity-weighted centroid. Pass 1 takes the global centroid;
    pass 2 re-centroids inside a window around the pass-1 estimate so
    bright asymmetric peaks outside the central region cannot bias the
    result.

    Bad pixels must already be handled by the caller (e.g. via
    :func:`inpaint_bad_pixels`) before invoking this function.

    Falls back to the geometric center if every pixel is rejected as noise.
    """
    height, width = pattern.shape[-2:]
    geometric_center = BeamCenter(x_px=width // 2, y_px=height // 2)

    working_pattern = pattern.astype(numpy.float64, copy=True)
    filtered_pattern = ndimage.median_filter(working_pattern, size=3)

    # Convert intensities to non-negative weights by subtracting the background
    # and rejecting pixels below background + mad_threshold * MAD. Otsu's
    # threshold is used to identify the background pool when the histogram is
    # bimodal; for unimodal noise-only inputs the helper falls back to
    # median/MAD over the whole frame.
    robust_statistics = estimate_noise_floor(filtered_pattern)
    background_intensity = robust_statistics.median
    significance_threshold = robust_statistics.get_significance_threshold(mad_threshold)
    centroid_weights = filtered_pattern - background_intensity
    centroid_weights[filtered_pattern < significance_threshold] = 0.0
    numpy.clip(centroid_weights, 0.0, None, out=centroid_weights)

    # Pixel-coordinate axes centered on the array midpoint, so symmetric weight
    # distributions sum to ~0 rather than relying on catastrophic cancellation.
    midpoint_y = (height - 1) / 2.0
    midpoint_x = (width - 1) / 2.0
    centered_y_axis = numpy.arange(height, dtype=numpy.float64).reshape(-1, 1) - midpoint_y
    centered_x_axis = numpy.arange(width, dtype=numpy.float64).reshape(1, -1) - midpoint_x

    # Pass 1: global intensity-weighted centroid.
    coarse_total_weight = float(centroid_weights.sum())

    if coarse_total_weight <= 0.0:
        return geometric_center

    coarse_center_y = midpoint_y + float(
        (centroid_weights * centered_y_axis).sum() / coarse_total_weight
    )
    coarse_center_x = midpoint_x + float(
        (centroid_weights * centered_x_axis).sum() / coarse_total_weight
    )

    # Pass 2: re-centroid inside a square window around the pass-1 estimate.
    half_window_size = min(height, width) // 4
    pixel_y = numpy.arange(height).reshape(-1, 1)
    pixel_x = numpy.arange(width).reshape(1, -1)
    in_central_window = (numpy.abs(pixel_y - coarse_center_y) <= half_window_size) & (
        numpy.abs(pixel_x - coarse_center_x) <= half_window_size
    )
    windowed_weights = centroid_weights * in_central_window
    refined_total_weight = float(windowed_weights.sum())

    if refined_total_weight > 0.0:
        refined_center_y = midpoint_y + float(
            (windowed_weights * centered_y_axis).sum() / refined_total_weight
        )
        refined_center_x = midpoint_x + float(
            (windowed_weights * centered_x_axis).sum() / refined_total_weight
        )
    else:
        refined_center_y = coarse_center_y
        refined_center_x = coarse_center_x

    return BeamCenter(
        x_px=int(round(refined_center_x)),
        y_px=int(round(refined_center_y)),
    )


@dataclass(frozen=True)
class SignalRadius:
    """How far from the beam center a diffraction pattern rises above its noise floor.

    `radius_px` is the outermost ring whose background-subtracted mean exceeds
    `threshold`, or 0 when none does. `radial_profile` holds those means, one per
    one-pixel-wide ring indexed by its radius, NaN where a ring had too few good pixels
    to be judged.
    """

    radius_px: int
    radial_profile: RealArrayType
    noise_floor: RobustStatistics
    threshold: float


def estimate_signal_radius(
    pattern: DiffractionPattern,
    beam_center: BeamCenter,
    *,
    bad_pixels: BadPixels | None = None,
    mad_threshold: float = 4.5,
    min_ring_pixels: int = 16,
) -> SignalRadius:
    """Estimate the radius about `beam_center` beyond which the signal is below the noise floor.

    Pipeline: 3x3 median filter -> Otsu-based noise-floor estimate over the good pixels
    -> mean of each one-pixel-wide ring, less the background -> the outermost ring whose
    mean exceeds ``mad_threshold * MAD``. Taking the outermost ring rather than the first
    one to fall below the threshold steps over a beamstop shadow near the center and
    keeps isolated high-angle rings, which are signal. Rings run out to the farthest
    detector corner, so a beam near an edge is judged on the partial rings it has.

    Corrections are the caller's, as for :func:`estimate_beam_center`: pass a pattern
    with background and flat field already applied and interior bad pixels repaired.
    Pixels flagged in `bad_pixels` -- typically the border-touching regions
    :func:`inpaint_bad_pixels` zeroes rather than fills -- are left out of every ring.
    The radius is in the pattern's own pixels.

    ``mad_threshold`` scales the noise MAD into the threshold a ring mean must exceed;
    larger values report a smaller radius. A ring with fewer than ``min_ring_pixels``
    good pixels is not judged, so a corner sliver or a ring crossing a detector gap
    cannot set the radius. When the noise MAD is zero, as on a mostly-empty
    photon-counting frame, any ring with a mean above the background counts as signal.
    """
    if mad_threshold < 0.0:
        raise ValueError(f'mad_threshold must be non-negative, got {mad_threshold}.')

    height, width = pattern.shape[-2:]
    filtered_pattern = ndimage.median_filter(pattern.astype(numpy.float64), size=3)
    good_pixels = (
        numpy.ones((height, width), dtype=bool)
        if bad_pixels is None
        else numpy.logical_not(bad_pixels)
    )

    if not numpy.any(good_pixels):
        raise ValueError('Every pixel is bad, so no noise floor can be estimated.')

    good_values = filtered_pattern[good_pixels]
    noise_floor = estimate_noise_floor(good_values)
    threshold = mad_threshold * noise_floor.median_absolute_deviation

    pixel_y = numpy.arange(height).reshape(-1, 1) - beam_center.y_px
    pixel_x = numpy.arange(width).reshape(1, -1) - beam_center.x_px
    ring_index = numpy.hypot(pixel_x, pixel_y).astype(numpy.intp)[good_pixels]

    num_rings = int(ring_index.max()) + 1 if ring_index.size > 0 else 0
    ring_sums = numpy.bincount(ring_index, weights=good_values, minlength=num_rings)
    ring_counts = numpy.bincount(ring_index, minlength=num_rings)

    radial_profile = numpy.full(num_rings, numpy.nan)
    judged = ring_counts >= min_ring_pixels
    radial_profile[judged] = ring_sums[judged] / ring_counts[judged] - noise_floor.median

    # NaN compares False, so unjudged rings never count as signal.
    (signal_rings,) = numpy.nonzero(radial_profile > threshold)

    return SignalRadius(
        radius_px=int(signal_rings[-1]) if signal_rings.size > 0 else 0,
        radial_profile=radial_profile,
        noise_floor=noise_floor,
        threshold=float(threshold),
    )
