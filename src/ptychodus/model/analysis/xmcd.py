from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectArrayType

from ..product import ObjectRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XMCDResult:
    pixel_width_m: float
    pixel_height_m: float
    center_x_m: float
    center_y_m: float
    polar_difference: ObjectArrayType
    polar_sum: ObjectArrayType
    polar_ratio: ObjectArrayType

    @property
    def pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(self.pixel_width_m, self.pixel_height_m)


class XMCDAnalyzer:
    # TODO feature request: want ability to align/add reconstructed slices
    #      of repeat scans for each polarization separately to improve statistics

    def __init__(self, repository: ObjectRepository) -> None:
        self._repository = repository

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> XMCDResult:
        lcircObject = self._repository[lcircItemIndex].getObject()
        rcircObject = self._repository[rcircItemIndex].getObject()

        if lcircObject.widthInPixels != rcircObject.widthInPixels:
            raise ValueError('Object width mismatch!')

        if lcircObject.heightInPixels != rcircObject.heightInPixels:
            raise ValueError('Object height mismatch!')

        if lcircObject.pixelWidthInMeters != rcircObject.pixelWidthInMeters:
            raise ValueError('Object pixel width mismatch!')

        if lcircObject.pixelHeightInMeters != rcircObject.pixelHeightInMeters:
            raise ValueError('Object pixel height mismatch!')

        # TODO align lcircArray/rcircArray

        lcircAmp = numpy.absolute(lcircObject.array)
        rcircAmp = numpy.absolute(rcircObject.array)

        ratio = numpy.divide(lcircAmp, rcircAmp)
        product = numpy.multiply(lcircAmp, rcircAmp)

        polar_difference = numpy.log(ratio, out=numpy.zeros_like(ratio), where=(ratio > 0))
        polar_sum = numpy.log(product, out=numpy.zeros_like(product), where=(product > 0))
        polar_ratio = numpy.divide(
            polar_difference,
            polar_sum,
            out=numpy.zeros_like(polar_sum),
            where=(polar_sum > 0),
        )

        return XMCDResult(
            pixel_width_m=rcircObject.pixelWidthInMeters,
            pixel_height_m=rcircObject.pixelHeightInMeters,
            center_x_m=rcircObject.centerXInMeters,
            center_y_m=rcircObject.centerYInMeters,
            polar_difference=polar_difference,
            polar_sum=polar_sum,
            polar_ratio=polar_ratio,
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def saveResult(self, filePath: Path, result: XMCDResult) -> None:
        numpy.savez(
            filePath,
            'pixel_height_m',
            result.pixel_height_m,
            'pixel_width_m',
            result.pixel_width_m,
            'center_x_m',
            result.center_x_m,
            'center_y_m',
            result.center_y_m,
            'polar_difference',
            result.polar_difference,
            'polar_sum',
            result.polar_sum,
            'polar_ratio',
            result.polar_ratio,
        )
