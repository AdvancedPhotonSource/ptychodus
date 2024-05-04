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
class DichroicResult:
    pixelGeometry: PixelGeometry
    polarDifference: ObjectArrayType
    polarSum: ObjectArrayType
    polarRatio: ObjectArrayType


class DichroicAnalyzer:
    # TODO feature request: want ability to align/add reconstructed slices
    #      of repeat scans for each polarization separately to improve statistics

    def __init__(self, repository: ObjectRepository) -> None:
        self._repository = repository

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> DichroicResult:
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

        polarDifference = numpy.log(ratio, out=numpy.zeros_like(ratio), where=(ratio > 0))
        polarSum = numpy.log(product, out=numpy.zeros_like(product), where=(product > 0))
        polarRatio = numpy.divide(polarDifference,
                                  polarSum,
                                  out=numpy.zeros_like(polarSum),
                                  where=(polarSum > 0))

        return DichroicResult(
            pixelGeometry=rcircObject.getPixelGeometry(),
            polarDifference=polarDifference,
            polarSum=polarSum,
            polarRatio=polarRatio,
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def saveResult(self, filePath: Path, result: DichroicResult) -> None:
        numpy.savez(
            filePath,
            'pixel_height_m',
            result.pixelGeometry.heightInMeters,
            'pixel_width_m',
            result.pixelGeometry.widthInMeters,
            'polar_difference',
            result.polarDifference,
            'polar_sum',
            result.polarSum,
            'polar_ratio',
            result.polarRatio,
        )
