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
        if self.lower_bound is not None:
            data[data < self.lower_bound] = 0

        if self.upper_bound is not None:
            data[data >= self.upper_bound] = 0

        return data


class DiffractionPatternCrop:
    def __init__(self, center: CropCenter, extent: ImageExtent) -> None:
        center_x = center.position_x_px
        radius_x = extent.width_px // 2
        self.slice_x = slice(center_x - radius_x, center_x + radius_x)

        center_y = center.position_y_px
        radius_y = extent.height_px // 2
        self.slice_y = slice(center_y - radius_y, center_y + radius_y)

    def apply_bool(self, data: BadPixels) -> BadPixels:
        return data[self.slice_y, self.slice_x]

    def apply(self, data: DiffractionPatterns) -> DiffractionPatterns:
        return data[:, self.slice_y, self.slice_x]


@dataclass(frozen=True)
class DiffractionPatternBinning:
    bin_size_x: int
    bin_size_y: int

    def apply_bool(self, data: BadPixels) -> BadPixels:
        binned_width = data.shape[-1] // self.bin_size_x
        binned_height = data.shape[-2] // self.bin_size_y
        shape = (binned_height, self.bin_size_y, binned_width, self.bin_size_x)
        return numpy.logical_and.reduce(data.reshape(shape), axis=(-3, -1), keepdims=False)

    def apply(self, data: DiffractionPatterns) -> DiffractionPatterns:
        binned_width = data.shape[-1] // self.bin_size_x
        binned_height = data.shape[-2] // self.bin_size_y
        shape = (-1, binned_height, self.bin_size_y, binned_width, self.bin_size_x)
        return numpy.sum(data.reshape(shape), axis=(-3, -1), keepdims=False)


@dataclass(frozen=True)
class DiffractionPatternPadding:
    pad_x: int
    pad_y: int

    def apply_bool(self, data: BadPixels) -> BadPixels:
        pad_width = (self.pad_y, self.pad_y, self.pad_x, self.pad_x)
        return numpy.pad(data, pad_width, mode='constant', constant_values=False)

    def apply(self, data: DiffractionPatterns) -> DiffractionPatterns:
        pad_width = (0, 0, self.pad_y, self.pad_y, self.pad_x, self.pad_x)
        return numpy.pad(data, pad_width, mode='constant', constant_values=0)


@dataclass(frozen=True)
class DiffractionPatternProcessor:
    crop: DiffractionPatternCrop | None
    filter_values: DiffractionPatternFilterValues | None
    binning: DiffractionPatternBinning | None
    padding: DiffractionPatternPadding | None
    hflip: bool
    vflip: bool
    transpose: bool

    def process_bad_pixels(self, bad_pixels: BadPixels) -> BadPixels:
        if bad_pixels.ndim != 2:
            raise ValueError(f'Invalid bad_pixel dimensions! (shape={bad_pixels.shape})')

        processed_bad_pixels = bad_pixels.copy()

        if self.crop is not None:
            processed_bad_pixels = self.crop.apply_bool(processed_bad_pixels)

        if self.binning is not None:
            processed_bad_pixels = self.binning.apply_bool(processed_bad_pixels)

        if self.padding is not None:
            processed_bad_pixels = self.padding.apply_bool(processed_bad_pixels)

        if self.hflip:
            processed_bad_pixels = numpy.flip(processed_bad_pixels, axis=-1)

        if self.vflip:
            processed_bad_pixels = numpy.flip(processed_bad_pixels, axis=-2)

        if self.transpose:
            processed_bad_pixels = numpy.transpose(processed_bad_pixels, axes=(0, 2, 1))

        return processed_bad_pixels

    def __call__(self, array: DiffractionArray) -> DiffractionArray:
        data = array.get_patterns()

        if data.ndim == 2:
            data = data[numpy.newaxis, ...]
        elif data.ndim != 3:
            raise ValueError(f'Invalid diffraction pattern dimensions! (shape={data.shape})')

        if self.filter_values is not None:
            data = self.filter_values.apply(data)

        if self.crop is not None:
            data = self.crop.apply(data)

        if self.binning is not None:
            data = self.binning.apply(data)

        if self.padding is not None:
            data = self.padding.apply(data)

        if self.hflip:
            data = numpy.flip(data, axis=-1)

        if self.vflip:
            data = numpy.flip(data, axis=-2)

        if self.transpose:
            data = numpy.transpose(data, axes=(0, 2, 1))

        return SimpleDiffractionArray(array.get_label(), array.get_indexes(), data)
