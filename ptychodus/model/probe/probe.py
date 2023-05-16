from decimal import Decimal
import logging

import numpy

from ...api.image import ImageExtent
from ...api.observer import Observable
from ...api.probe import ProbeArrayType
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


class Probe(Observable):  # FIXME

    def __init__(self, sizer: ProbeSizer) -> None:
        super().__init__()
        self._array = numpy.zeros((1, *sizer.getProbeExtent().shape), dtype=complex)

    def getProbeExtent(self) -> ImageExtent:
        return ImageExtent(width=self._array.shape[-1], height=self._array.shape[-2])

    def getNumberOfProbeModes(self) -> int:
        return self._array.shape[0]

    def getProbeMode(self, index: int) -> ProbeArrayType:
        return self._array[index, ...]

    def getProbeModeRelativePower(self, index: int) -> Decimal:
        if numpy.isnan(self._array).any():
            logger.error('Probe contains NaN value(s)!')
            return Decimal()

        probe = self._array
        power = numpy.sum((probe * probe.conj()).real, axis=(-2, -1))
        powersum = power.sum()

        if powersum > 0.:
            power /= powersum

        return Decimal(repr(power[index]))

    def getArray(self) -> ProbeArrayType:
        return self._array

    def setArray(self, array: ProbeArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Probe must be a complex-valued ndarray')

        if array.ndim == 2:
            self._array = array[numpy.newaxis, ...]
        elif array.ndim == 3:
            self._array = array
        else:
            raise ValueError('Probe must be 2- or 3-dimensional ndarray.')

        self.notifyObservers()
