from __future__ import annotations
from dataclasses import dataclass

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    BadPixels,
    CropCenter,
    DiffractionArray,
    DiffractionPatterns,
    SimpleDiffractionArray,
)


@dataclass(frozen=True)
class DiffractionPatternFilterValues:
    lower_bound: int | None
    upper_bound: int | None

    def apply(self, data: DiffractionPatterns) -> DiffractionPatterns:
        if self.lower_bound is None and self.upper_bound is None:
            return data

        out = data.copy()

        if self.lower_bound is not None:
            out[out < self.lower_bound] = 0

        if self.upper_bound is not None:
            out[out >= self.upper_bound] = 0

        return out


class DiffractionPatternCrop:
    def __init__(self, center: CropCenter, extent: ImageExtent) -> None:
        center_x = center.position_x_px
        radius_x = extent.width_px // 2
        self.slice_x = slice(center_x - radius_x, center_x + radius_x)

        center_y = center.position_y_px
        radius_y = extent.height_px // 2
        self.slice_y = slice(center_y - radius_y, center_y + radius_y)

    def apply(self, data: numpy.ndarray, *, is_mask: bool = False) -> numpy.ndarray:
        leading = (slice(None),) * (data.ndim - 2)
        return data[(*leading, self.slice_y, self.slice_x)]


@dataclass(frozen=True)
class DiffractionPatternBinning:
    bin_size_x: int
    bin_size_y: int

    def apply(self, data: numpy.ndarray, *, is_mask: bool = False) -> numpy.ndarray:
        binned_height = data.shape[-2] // self.bin_size_y
        binned_width = data.shape[-1] // self.bin_size_x
        shape = data.shape[:-2] + (binned_height, self.bin_size_y, binned_width, self.bin_size_x)
        reshaped = data.reshape(shape)
        if is_mask:
            return numpy.logical_and.reduce(reshaped, axis=(-3, -1), keepdims=False)
        return numpy.sum(reshaped, axis=(-3, -1), keepdims=False)


@dataclass(frozen=True)
class DiffractionPatternPadding:
    pad_x: int
    pad_y: int

    def apply(self, data: numpy.ndarray, *, is_mask: bool = False) -> numpy.ndarray:
        leading_pad = ((0, 0),) * (data.ndim - 2)
        pad_width = (*leading_pad, (self.pad_y, self.pad_y), (self.pad_x, self.pad_x))
        fill = False if is_mask else 0
        return numpy.pad(data, pad_width, mode='constant', constant_values=fill)


@dataclass(frozen=True)
class DiffractionPatternProcessor:
    crop: DiffractionPatternCrop | None
    filter_values: DiffractionPatternFilterValues | None
    binning: DiffractionPatternBinning | None
    padding: DiffractionPatternPadding | None
    hflip: bool
    vflip: bool
    transpose: bool

    def _apply_geometric(self, data: numpy.ndarray, *, is_mask: bool) -> numpy.ndarray:
        """Run the geometric pipeline (crop → bin → pad → flips → transpose) on a 2-D mask
        or a 3-D pattern stack. Order matches __call__; mirror changes in both paths."""
        if self.crop is not None:
            data = self.crop.apply(data, is_mask=is_mask)

        if self.binning is not None:
            data = self.binning.apply(data, is_mask=is_mask)

        if self.padding is not None:
            data = self.padding.apply(data, is_mask=is_mask)

        if self.hflip:
            data = numpy.flip(data, axis=-1)

        if self.vflip:
            data = numpy.flip(data, axis=-2)

        if self.transpose:
            axes = tuple(range(data.ndim - 2)) + (data.ndim - 1, data.ndim - 2)
            data = numpy.transpose(data, axes=axes)

        return data

    def process_bad_pixels(self, bad_pixels: BadPixels) -> BadPixels:
        if bad_pixels.ndim != 2:
            raise ValueError(f'Invalid bad_pixel dimensions! (shape={bad_pixels.shape})')

        return self._apply_geometric(bad_pixels, is_mask=True)

    def __call__(self, array: DiffractionArray) -> DiffractionArray:
        patterns = array.get_patterns()

        if patterns.ndim == 2:
            patterns = patterns[numpy.newaxis, ...]
        elif patterns.ndim != 3:
            raise ValueError(f'Invalid diffraction pattern dimensions! (shape={patterns.shape})')

        if self.filter_values is not None:
            patterns = self.filter_values.apply(patterns)

        patterns = self._apply_geometric(patterns, is_mask=False)

        return SimpleDiffractionArray(array.get_label(), array.get_indexes(), patterns)
