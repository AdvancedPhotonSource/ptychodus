from decimal import Decimal

import numpy

from ...api.observer import Observable
from ...api.probe import ProbeArrayType
from .sizer import ProbeSizer


class Probe(Observable):

    def __init__(self, sizer: ProbeSizer) -> None:
        super().__init__()
        self._array = numpy.zeros((1, *sizer.getProbeExtent().shape), dtype=complex)

    def getNumberOfProbeModes(self) -> int:
        return self._array.shape[0]

    def getProbeMode(self, index: int) -> ProbeArrayType:
        return self._array[index, ...]

    def getProbeModeRelativePower(self, index: int) -> Decimal:
        # FIXME handle NaN in array
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
