from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectArrayType, ObjectCenter

from ..product import ObjectRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XMCDResult:
    pixel_geometry: PixelGeometry | None
    center: ObjectCenter | None
    polar_difference: ObjectArrayType
    polar_sum: ObjectArrayType
    polar_ratio: ObjectArrayType


class XMCDAnalyzer:
    # TODO feature request: want ability to align/add reconstructed slices
    #      of repeat scans for each polarization separately to improve statistics

    def __init__(self, repository: ObjectRepository) -> None:
        self._repository = repository

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> XMCDResult:
        lcircObject = self._repository[lcircItemIndex].getObject()
        rcircObject = self._repository[rcircItemIndex].getObject()

        lcircObjectGeometry = lcircObject.getGeometry()
        rcircObjectGeometry = rcircObject.getGeometry()

        if lcircObjectGeometry.widthInPixels != rcircObjectGeometry.widthInPixels:
            raise ValueError('Object width mismatch!')

        if lcircObjectGeometry.heightInPixels != rcircObjectGeometry.heightInPixels:
            raise ValueError('Object height mismatch!')

        if lcircObjectGeometry.pixelWidthInMeters != rcircObjectGeometry.pixelWidthInMeters:
            raise ValueError('Object pixel width mismatch!')

        if lcircObjectGeometry.pixelHeightInMeters != rcircObjectGeometry.pixelHeightInMeters:
            raise ValueError('Object pixel height mismatch!')

        # TODO align lcircArray/rcircArray

        # FIXME OPR
        lcircAmp = numpy.absolute(lcircObject.getArray())
        rcircAmp = numpy.absolute(rcircObject.getArray())

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
            pixel_geometry=rcircObject.getPixelGeometry(),
            center=rcircObject.getCenter(),
            polar_difference=polar_difference,
            polar_sum=polar_sum,
            polar_ratio=polar_ratio,
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def saveResult(self, file_path: Path, result: XMCDResult) -> None:
        contents: dict[str, Any] = {
            'polar_difference': result.polar_difference,
            'polar_sum': result.polar_sum,
            'polar_ratio': result.polar_ratio,
        }

        pixel_geometry = result.pixel_geometry

        if pixel_geometry is not None:
            contents['pixel_height_m'] = pixel_geometry.heightInMeters
            contents['pixel_width_m'] = pixel_geometry.widthInMeters

        center = result.center

        if center is not None:
            contents['center_x_m'] = center.positionXInMeters
            contents['center_y_m'] = center.positionYInMeters

        numpy.savez(file_path, **contents)
