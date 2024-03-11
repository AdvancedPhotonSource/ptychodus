from __future__ import annotations
from dataclasses import dataclass
import logging

import numpy

from ...api.object import ObjectArrayType
from ...api.patterns import PixelGeometry
from ..product import ObjectRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DichroicAnalysis:
    pixelGeometry: PixelGeometry
    polarDifference: ObjectArrayType
    polarSum: ObjectArrayType

    @property
    def polarRatio(self) -> ObjectArrayType:
        return numpy.divide(self.polarDifference,
                            self.polarSum,
                            out=numpy.zeros_like(self.polarSum),
                            where=(self.polarSum > 0.))


class DichroicAnalyzer:
    # TODO feature request: want ability to align/add reconstructed slices of
    #      repeat scans for each polarization separately to improve statistics

    def __init__(self, repository: ObjectRepository) -> None:
        self._repository = repository

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> DichroicAnalysis:
        lcircObject = self._repository[lcircItemIndex].getObject()
        rcircObject = self._repository[rcircItemIndex].getObject()

        # TODO geometry checks
        pixelGeometry = rcircObject.getPixelGeometry()
        # TODO align lcircArray/rcircArray

        lcircAmp = numpy.absolute(lcircObject.array)
        rcircAmp = numpy.absolute(rcircObject.array)

        lcircLogAmp = numpy.log(lcircAmp, out=numpy.zeros_like(lcircAmp), where=(lcircAmp > 0.))
        rcircLogAmp = numpy.log(rcircAmp, out=numpy.zeros_like(rcircAmp), where=(rcircAmp > 0.))

        return DichroicAnalysis(
            pixelGeometry=pixelGeometry,
            polarDifference=lcircLogAmp - rcircLogAmp,
            polarSum=lcircLogAmp + rcircLogAmp,
        )
