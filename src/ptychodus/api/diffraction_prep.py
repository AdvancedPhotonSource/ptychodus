"""Diffraction pattern preprocessing pipeline.

Steps share a single `apply(data) -> ndarray` interface and infer whether the
input is a 3-D pattern stack or a 2-D boolean bad-pixel mask from
`data.dtype`. The order of operations is encoded in
`DiffractionPrepPipeline.steps`; the canonical order emitted by the model-layer
factory is filter → crop → binning → padding → hflip → vflip → transpose.

Every model is a frozen Pydantic BaseModel so pipelines serialize to JSON for
reuse by the REST/MCP surface in `ptychodus_store`.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Annotated, Literal, TypeAlias

import numpy
from pydantic import BaseModel, ConfigDict, Discriminator, Field

from .diffraction import (
    BadPixels,
    CropCenter,
    DiffractionArray,
    DiffractionPatterns,
    SimpleDiffractionArray,
)
from .geometry import ImageExtent, PixelGeometry


def _is_mask(data: numpy.ndarray) -> bool:
    return data.dtype == numpy.bool_


class DiffractionPrepStep(BaseModel):
    """Abstract base for a single preprocessing step."""

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


class CropStep(DiffractionPrepStep):
    """Center-crop the last two axes to `extent` about `center` (pixel coords)."""

    type: Literal['crop'] = 'crop'
    center: CropCenter
    extent: ImageExtent

    def apply(self, data: numpy.ndarray) -> numpy.ndarray:
        radius_x = self.extent.width_px // 2
        slice_x = slice(self.center.position_x_px - radius_x, self.center.position_x_px + radius_x)
        radius_y = self.extent.height_px // 2
        slice_y = slice(self.center.position_y_px - radius_y, self.center.position_y_px + radius_y)
        leading = (slice(None),) * (data.ndim - 2)
        return data[(*leading, slice_y, slice_x)]

    def apply_to_extent(self, extent: ImageExtent) -> ImageExtent:
        return self.extent


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
        return numpy.sum(reshaped, axis=(-3, -1), keepdims=False)

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


DiffractionPrepStepUnion: TypeAlias = Annotated[
    FilterValuesStep
    | CropStep
    | BinningStep
    | PaddingStep
    | HorizontalFlipStep
    | VerticalFlipStep
    | TransposeStep,
    Discriminator('type'),
]


class DiffractionPrepPipeline(BaseModel):
    """Ordered chain of preprocessing steps applied to patterns and the bad-pixel mask."""

    model_config = ConfigDict(frozen=True)

    steps: tuple[DiffractionPrepStepUnion, ...] = ()

    def _apply(self, data: numpy.ndarray) -> numpy.ndarray:
        for step in self.steps:
            data = step.apply(data)
        return data

    def apply_to_patterns(self, patterns: DiffractionPatterns) -> DiffractionPatterns:
        if patterns.ndim == 2:
            patterns = patterns[numpy.newaxis, ...]
        elif patterns.ndim != 3:
            raise ValueError(f'Invalid diffraction pattern dimensions! (shape={patterns.shape})')

        return self._apply(patterns)

    def apply_to_mask(self, bad_pixels: BadPixels) -> BadPixels:
        if bad_pixels.ndim != 2:
            raise ValueError(f'Invalid bad_pixel dimensions! (shape={bad_pixels.shape})')

        return self._apply(bad_pixels)

    def __call__(self, array: DiffractionArray) -> DiffractionArray:
        patterns = self.apply_to_patterns(array.get_patterns())
        return SimpleDiffractionArray(array.get_label(), array.get_indexes(), patterns)

    def compute_output_extent(self, extent: ImageExtent) -> ImageExtent:
        for step in self.steps:
            extent = step.apply_to_extent(extent)
        return extent

    def compute_output_pixel_geometry(self, geometry: PixelGeometry) -> PixelGeometry:
        for step in self.steps:
            geometry = step.apply_to_pixel_geometry(geometry)
        return geometry
