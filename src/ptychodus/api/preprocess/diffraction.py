"""Diffraction pattern preprocessing: the step pipeline and standalone helpers.

Steps share a single `apply(data) -> ndarray` interface and infer whether the
input is a 3-D pattern stack or a 2-D boolean bad-pixel mask from
`data.dtype`. The order of operations is encoded in
`DiffractionPrepPipeline.steps`; the canonical order emitted by the model-layer
factory is filter → binning → upsample → padding → hflip → vflip → transpose.
Cropping is hoisted to load time (see :class:`DiffractionPrepPlan`) so readers
that support partial reads (HDF5) only fetch the crop rectangle from disk.

The free functions at the end of the module are helpers used *alongside* the
pipeline rather than steps within it, and they deliberately stay that way.
`zero_bad_pixels` runs in raw detector coordinates before the pipeline, and
`inpaint_bad_pixels` and `estimate_beam_center` take a whole frame at once. A
step that repaired or dropped frames would also break the index/pattern 1:1
invariant that `prepare_reconstruct_input` relies on, because
`DiffractionPrepPipeline.__call__` passes the original indexes through
unchanged.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

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
    DiffractionPatterns,
    SimpleDiffractionArray,
)
from ..geometry import ImageExtent, PixelGeometry
from .noise import estimate_noise_floor


def _is_mask(data: numpy.ndarray) -> bool:
    return data.dtype == numpy.bool_


class DiffractionPrepStep(BaseModel):
    """Abstract base for a single preprocessing step.

    To add a new step:

    - Declare `type: Literal['<unique_tag>'] = '<unique_tag>'`. The default is required so
      callers never pass the tag and `model_dump_json` emits it automatically.
    - Add the class to `DiffractionPrepStepUnion`, or it is unreachable on deserialize.
    - Override `apply_to_extent` / `apply_to_pixel_geometry` / `apply_to_dtype` only where the
      step is not identity in that dimension; the defaults below pass the input through unchanged.
    - Branch on `_is_mask(data)` in `apply` if pattern and mask behavior differ, as
      `BinningStep` does (sum for patterns, logical-AND for masks).
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        """Apply this step. Mask vs pattern behavior is inferred from `data.dtype`."""

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        """Return the extent this step would produce given `extent`. Default: identity."""
        return extent

    def apply_to_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        """Return the pixel geometry this step would produce. Default: identity."""
        return geometry

    def apply_to_dtype(self, dtype: numpy.dtype) -> numpy.dtype:
        """Return the dtype this step would produce given `dtype`. Default: identity."""
        return dtype


class FilterValuesStep(DiffractionPrepStep):
    """Zero pattern values outside `[lower_bound, upper_bound)`. No-op on masks."""

    type: Literal['filter_values'] = 'filter_values'
    lower_bound: int | None = None
    upper_bound: int | None = None

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
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


class BinningStep(DiffractionPrepStep):
    """Reduce each `bin_size_y × bin_size_x` block. Sum for patterns; logical-AND for masks."""

    type: Literal['binning'] = 'binning'
    bin_size_x: int = Field(gt=0)
    bin_size_y: int = Field(gt=0)

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        binned_height = data.shape[-2] // self.bin_size_y
        binned_width = data.shape[-1] // self.bin_size_x
        shape = data.shape[:-2] + (binned_height, self.bin_size_y, binned_width, self.bin_size_x)
        reshaped = data.reshape(shape)
        if _is_mask(data):
            return numpy.logical_and.reduce(reshaped, axis=(-3, -1), keepdims=False)
        return numpy.sum(
            reshaped, axis=(-3, -1), keepdims=False, dtype=self.apply_to_dtype(data.dtype)
        )

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

    def apply_to_dtype(self, dtype: numpy.dtype) -> numpy.dtype:
        if not numpy.issubdtype(dtype, numpy.integer):
            return dtype
        factor = self.bin_size_x * self.bin_size_y
        if factor <= 1:
            return dtype
        needed_max = int(numpy.iinfo(dtype).max) * factor
        return numpy.promote_types(dtype, numpy.min_scalar_type(needed_max))


class UpsampleStep(DiffractionPrepStep):
    """FFT zero-pad upsampling by an isotropic integer factor.

    Patterns: band-limited (sinc) interpolation via FFT zero-padding, scaled by
    ``factor**2`` to preserve per-pixel intensity semantics. Residual sinc-ringing
    undershoots are floored to each pattern's pre-upsample minimum.

    Bad-pixel mask: Kronecker tile via ``numpy.repeat`` — exact for integer factors
    and equivalent to nearest-neighbor zoom on booleans.
    """

    type: Literal['upsample'] = 'upsample'
    factor: int = Field(gt=0)

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        if self.factor == 1:
            return data

        if _is_mask(data):
            return numpy.repeat(numpy.repeat(data, self.factor, axis=-2), self.factor, axis=-1)

        pattern_min = data.min(axis=(-2, -1), keepdims=True)
        h, w = data.shape[-2:]
        new_h, new_w = h * self.factor, w * self.factor

        spectrum = numpy.fft.fftshift(numpy.fft.fft2(data, axes=(-2, -1)), axes=(-2, -1))
        padded = numpy.zeros(data.shape[:-2] + (new_h, new_w), dtype=spectrum.dtype)
        pad_y, pad_x = (new_h - h) // 2, (new_w - w) // 2
        padded[..., pad_y : pad_y + h, pad_x : pad_x + w] = spectrum
        upsampled = numpy.fft.ifft2(numpy.fft.ifftshift(padded, axes=(-2, -1)), axes=(-2, -1)).real
        upsampled *= self.factor * self.factor

        return numpy.clip(upsampled, pattern_min, None).astype(data.dtype, copy=False)

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        return ImageExtent(
            width_px=extent.width_px * self.factor,
            height_px=extent.height_px * self.factor,
        )

    def apply_to_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        return PixelGeometry(
            width_m=geometry.width_m / self.factor,
            height_m=geometry.height_m / self.factor,
        )


class PaddingStep(DiffractionPrepStep):
    """Symmetrically pad the last two axes. Fill 0 for patterns; False for masks."""

    type: Literal['padding'] = 'padding'
    pad_x: int = Field(ge=0)
    pad_y: int = Field(ge=0)

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        leading_pad = ((0, 0),) * (data.ndim - 2)
        pad_width = (*leading_pad, (self.pad_y, self.pad_y), (self.pad_x, self.pad_x))
        fill = False if _is_mask(data) else 0
        return numpy.pad(data, pad_width, mode='constant', constant_values=fill)

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        return ImageExtent(
            width_px=extent.width_px + 2 * self.pad_x,
            height_px=extent.height_px + 2 * self.pad_y,
        )


class HorizontalFlipStep(DiffractionPrepStep):
    """Flip the last axis."""

    type: Literal['hflip'] = 'hflip'

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        return numpy.flip(data, axis=-1)


class VerticalFlipStep(DiffractionPrepStep):
    """Flip the second-to-last axis."""

    type: Literal['vflip'] = 'vflip'

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        return numpy.flip(data, axis=-2)


class TransposeStep(DiffractionPrepStep):
    """Swap the last two axes."""

    type: Literal['transpose'] = 'transpose'

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        axes = tuple(range(data.ndim - 2)) + (data.ndim - 1, data.ndim - 2)
        return numpy.transpose(data, axes=axes)

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        return ImageExtent(width_px=extent.height_px, height_px=extent.width_px)

    def apply_to_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        return PixelGeometry(width_m=geometry.height_m, height_m=geometry.width_m)


DiffractionPrepStepUnion: TypeAlias = (
    FilterValuesStep
    | BinningStep
    | UpsampleStep
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

    def _apply(self, data: numpy.ndarray) -> numpy.ndarray:
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

    def compute_output_dtype(self, dtype: numpy.dtype) -> numpy.dtype:
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


def inpaint_bad_pixels(
    pattern: DiffractionPattern, bad_pixels: BadPixels | None
) -> DiffractionPattern:
    """Return `pattern` with bad-pixel positions filled by biharmonic inpainting.

    Returns the input unchanged when `bad_pixels` is None or all-False so the
    (expensive) skimage call is skipped. When inpainting runs the result is
    float64; the input dtype is preserved when it does not.
    """
    if bad_pixels is None or not numpy.any(bad_pixels):
        return pattern

    return inpaint_biharmonic(pattern.astype(numpy.float64), bad_pixels)


def estimate_beam_center(
    pattern: DiffractionPattern,
    bad_pixels: BadPixels | None = None,
    *,
    mad_threshold: float = 4.5,
) -> BeamCenter:
    """Estimate the direct-beam pixel of a diffraction pattern.

    Two passes: pass 1 takes the global intensity-weighted center; pass 2
    re-centroids inside a window around the pass-1 estimate so bright
    asymmetric peaks outside the central region cannot bias the result.

    Falls back to the geometric center if all pixels are masked or rejected.
    """
    height, width = pattern.shape[-2:]
    geometric_center = BeamCenter(x_px=width // 2, y_px=height // 2)

    # Median-filter a float copy with bad pixels zeroed.
    working_pattern = pattern.astype(numpy.float64, copy=True)

    if bad_pixels is not None and numpy.any(bad_pixels):
        good_pixel_mask = numpy.logical_not(bad_pixels)
        working_pattern[bad_pixels] = 0.0
    else:
        good_pixel_mask = numpy.ones(pattern.shape[-2:], dtype=bool)

    filtered_pattern = ndimage.median_filter(working_pattern, size=3)

    # Convert intensities to non-negative weights by subtracting the background
    # and rejecting pixels below background + mad_threshold * MAD. Otsu's
    # threshold is used to identify the background pool when the histogram is
    # bimodal; for unimodal noise-only inputs the helper falls back to
    # median/MAD over all good pixels.
    good_pixel_intensities = filtered_pattern[good_pixel_mask]

    if good_pixel_intensities.size == 0:
        return geometric_center

    robust_statistics = estimate_noise_floor(good_pixel_intensities)
    background_intensity = robust_statistics.median
    significance_threshold = robust_statistics.get_significance_threshold(mad_threshold)
    centroid_weights = filtered_pattern - background_intensity
    centroid_weights[~good_pixel_mask] = 0.0
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
