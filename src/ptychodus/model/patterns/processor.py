from __future__ import annotations
from dataclasses import dataclass

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    BooleanArrayType,
    CropCenter,
    DiffractionPatternArray,
    DiffractionPatternArrayType,
    SimpleDiffractionPatternArray,
)


class DiffractionPatternCrop:
    def __init__(self, center: CropCenter, extent: ImageExtent) -> None:
        center_x = center.positionXInPixels
        radius_x = extent.widthInPixels // 2
        self.slice_x = slice(center_x - radius_x, center_x + radius_x)

        center_y = center.positionYInPixels
        radius_y = extent.heightInPixels // 2
        self.slice_y = slice(center_y - radius_y, center_y + radius_y)

    def apply(self, data: DiffractionPatternArrayType) -> DiffractionPatternArrayType:
        return data[:, self.slice_y, self.slice_x]


@dataclass(frozen=True)
class DiffractionPatternBinning:
    bin_size_x: int
    bin_size_y: int

    def apply(self, data: DiffractionPatternArrayType) -> DiffractionPatternArrayType:
        binned_width = data.shape[-1] // self.bin_size_x
        binned_height = data.shape[-2] // self.bin_size_y
        shape = (-1, binned_height, self.bin_size_y, binned_width, self.bin_size_x)
        return numpy.sum(data.reshape(shape), axis=(-3, -1), keepdims=False)


@dataclass(frozen=True)
class DiffractionPatternPadding:
    pad_x: int
    pad_y: int

    def apply(self, data: DiffractionPatternArrayType) -> DiffractionPatternArrayType:
        pad_width = (0, 0, self.pad_y, self.pad_y, self.pad_x, self.pad_x)
        return numpy.pad(data, pad_width, mode='constant', constant_values=0)


@dataclass(frozen=True)
class DiffractionPatternProcessor:
    bad_pixels: BooleanArrayType | None
    crop: DiffractionPatternCrop | None
    binning: DiffractionPatternBinning | None
    padding: DiffractionPatternPadding | None
    flip_x: bool
    flip_y: bool

    def __call__(self, array: DiffractionPatternArray) -> DiffractionPatternArray:
        data = array.getData()

        if data.ndim != 3:
            raise ValueError(f'Invalid diffraction pattern dimensions! (shape={data.shape})')

        if self.bad_pixels is not None:
            data[:, self.bad_pixels] = 0

        if self.crop is not None:
            data = self.crop.apply(data)

        if self.binning is not None:
            data = self.binning.apply(data)

        if self.padding is not None:
            data = self.padding.apply(data)

        if self.flip_x:
            data = numpy.flip(data, axis=-1)

        if self.flip_y:
            data = numpy.flip(data, axis=-2)

        return SimpleDiffractionPatternArray(
            array.getLabel(), array.getIndex(), data, array.getState()
        )
