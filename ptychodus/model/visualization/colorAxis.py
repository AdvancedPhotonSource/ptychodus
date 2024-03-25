import logging

import numpy

from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.visualization import NumberArrayType, RealArrayType

logger = logging.getLogger(__name__)


class DataArray:  # FIXME remove

    def __init__(self, array: NumberArrayType) -> None:
        self._array = array

    def getZeros(self, channels: int) -> RealArrayType:  # FIXME move
        return numpy.zeros((*self._array.shape, channels), dtype=numpy.single)


class ColorAxis(ParameterRepository):

    def __init__(self) -> None:
        super().__init__('color_axis')
        self.lower = self._registerRealParameter('lower', 0.)
        self.upper = self._registerRealParameter('upper', 1.)

    def setToDataRange(self, array: RealArrayType) -> None:
        lower = 0.
        upper = 1.

        if array.size > 0:
            lower = array.min()
            upper = array.max()

            if not (numpy.isfinite(lower) and numpy.isfinite(upper)):
                logger.warning('Array values not finite!')
                lower = 0.
                upper = 1.
            elif lower == upper:
                logger.debug('Array values are uniform.')
                lower -= 0.5
                upper += 0.5

        self.lower.setValue(lower, notify=False)
        self.upper.setValue(upper, notify=False)
        self.notifyObservers()

    def normalize(self, array: RealArrayType) -> RealArrayType:  # FIXME use or lose
        lower = self.lower.getValue()
        upper = self.upper.getValue()
        width = upper - lower

        if not numpy.isfinite(width):
            logger.warning('Color axis not finite!')
            return numpy.zeros_like(array)

        if width <= 0:
            logger.warning('Invalid color axis width!')
            return numpy.zeros_like(array)

        result = (array - lower) / width
        result[result < lower] = lower
        result[result > upper] = upper
        return result
