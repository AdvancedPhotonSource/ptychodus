from typing import Any, Union
import logging

from skimage.restoration import unwrap_phase
import numpy
import numpy.typing

from ...api.image import RealArrayType
from ...api.observer import Observable

NumericDTypes = Union[numpy.integer[Any], numpy.floating[Any], numpy.complexfloating[Any, Any]]
NumericArrayType = numpy.typing.NDArray[NumericDTypes]

logger = logging.getLogger(__name__)


class VisualizationArray(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._array: NumericArrayType = numpy.zeros((1, 1))

    def getRealPart(self) -> RealArrayType:
        return numpy.real(self._array).astype(numpy.float_)

    def getImaginaryPart(self) -> RealArrayType:
        return numpy.imag(self._array).astype(numpy.float_)

    def getAmplitude(self) -> RealArrayType:
        return numpy.absolute(self._array).astype(numpy.float_)

    def getPhaseInRadians(self) -> RealArrayType:
        return numpy.angle(self._array).astype(numpy.float_)

    def getPhaseUnwrappedInRadians(self) -> RealArrayType:
        return unwrap_phase(self.getPhaseInRadians())

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    def _reset(self) -> None:
        self._array = numpy.zeros((1, 1))

    def setArray(self, array: NumericArrayType) -> None:
        if array is None:
            logger.error('Refusing to assign null array!')
            self._reset()
        elif numpy.size(array) < 1:
            logger.error('Refusing to assign empty array!')
            self._reset()
        else:
            self._array = array

        self.notifyObservers()
