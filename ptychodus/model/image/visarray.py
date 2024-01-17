from typing import Any, Union
import logging

from skimage.restoration import unwrap_phase
import numpy
import numpy.typing

from ...api.observer import Observable
from ...api.patterns import PixelGeometry
from ...api.visualize import RealArrayType

NumericDTypes = Union[numpy.integer[Any], numpy.floating[Any], numpy.complexfloating[Any, Any]]
NumericArrayType = numpy.typing.NDArray[NumericDTypes]

logger = logging.getLogger(__name__)


class VisualizationArray(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._array: NumericArrayType = numpy.zeros((0, 0))
        self._pixelGeometry = PixelGeometry.createNull()

    def getRealPart(self) -> RealArrayType:
        return numpy.real(self._array).astype(numpy.float_)

    def getImaginaryPart(self) -> RealArrayType:
        return numpy.imag(self._array).astype(numpy.float_)

    def getAmplitude(self) -> RealArrayType:
        return numpy.absolute(self._array).astype(numpy.float_)

    def getIntensity(self) -> RealArrayType:
        if issubclass(self._array.dtype.type, numpy.integer):
            return self._array.astype(numpy.float_)
        else:
            return numpy.square(self.getAmplitude())

    def getPhaseInRadians(self) -> RealArrayType:
        return numpy.angle(self._array).astype(numpy.float_)

    def getPhaseUnwrappedInRadians(self) -> RealArrayType:
        return unwrap_phase(self.getPhaseInRadians())

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def size(self) -> int:
        return self._array.size

    @property
    def pixelGeometry(self) -> PixelGeometry:
        return self._pixelGeometry

    def clearArray(self) -> None:
        self._array = numpy.zeros((0, 0))
        self._pixelGeometry = PixelGeometry.createNull()
        self.notifyObservers()

    def setArray(self, array: NumericArrayType, pixelGeometry: PixelGeometry) -> None:
        self._array = array
        self._pixelGeometry = pixelGeometry
        self.notifyObservers()
