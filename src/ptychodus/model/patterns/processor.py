from __future__ import annotations
from dataclasses import dataclass

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    CropCenter,
    DiffractionPatternArray,
    PatternDataType,
    SimpleDiffractionPatternArray,
)


@dataclass(frozen=True)
class DiffractionPatternFilterValues:
    lower_bound: int | None
    upper_bound: int | None

    def apply(self, data: PatternDataType) -> PatternDataType:
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

    def apply(self, data: PatternDataType) -> PatternDataType:
        return data[:, self.slice_y, self.slice_x]


@dataclass(frozen=True)
class DiffractionPatternBinning:
    bin_size_x: int
    bin_size_y: int

    def apply(self, data: PatternDataType) -> PatternDataType:
        binned_width = data.shape[-1] // self.bin_size_x
        binned_height = data.shape[-2] // self.bin_size_y
        shape = (-1, binned_height, self.bin_size_y, binned_width, self.bin_size_x)
        return numpy.sum(data.reshape(shape), axis=(-3, -1), keepdims=False)


@dataclass(frozen=True)
class DiffractionPatternPadding:
    pad_x: int
    pad_y: int

    def apply(self, data: PatternDataType) -> PatternDataType:
        pad_width = (0, 0, self.pad_y, self.pad_y, self.pad_x, self.pad_x)
        return numpy.pad(data, pad_width, mode='constant', constant_values=0)


@dataclass(frozen=True)
class DiffractionPatternProcessor:
    crop: DiffractionPatternCrop | None
    filter_values: DiffractionPatternFilterValues | None
    binning: DiffractionPatternBinning | None
    padding: DiffractionPatternPadding | None
    flip_x: bool
    flip_y: bool

    def __call__(self, array: DiffractionPatternArray) -> DiffractionPatternArray:
        data = array.get_data()

        if data.ndim != 3:
            raise ValueError(f'Invalid diffraction pattern dimensions! (shape={data.shape})')

        if self.crop is not None:
            data = self.crop.apply(data)

        if self.filter_values is not None:
            data = self.filter_values.apply(data)

        if self.binning is not None:
            # TODO handle binning with bad pixels
            data = self.binning.apply(data)

        if self.padding is not None:
            data = self.padding.apply(data)

        if self.flip_y:
            data = numpy.flip(data, axis=-2)

        if self.flip_x:
            data = numpy.flip(data, axis=-1)

        return SimpleDiffractionPatternArray(array.get_label(), array.get_indexes(), data)
