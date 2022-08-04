from decimal import Decimal
import logging

import numpy

from ...api.observer import Observable
from ...api.probe import ProbeArrayType
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


class Probe(Observable):

    def __init__(self, sizer: ProbeSizer) -> None:
        super().__init__()
        self._array = numpy.zeros((1, *sizer.getProbeExtent().shape), dtype=complex)

    def getNumberOfProbeModes(self) -> int:
        return self._array.shape[0]

    def getProbeMode(self, index: int) -> ProbeArrayType:
        return self._array[index, ...]

    def getProbeModeRelativePower(self, index: int) -> Decimal:
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

    def addAMode(self) -> None:
        # randomly shift the first mode
        pw = self._array.shape[-1]

        variate1 = numpy.random.rand(2, 1) - 0.5  # FIXME rng
        variate2 = (numpy.arange(0, pw) + 0.5) / pw - 0.5
        variate = variate1 * variate2

        phaseShift = numpy.exp(-2j * numpy.pi * variate)
        mode = self._array[:1, :, :] * phaseShift[0][numpy.newaxis] * phaseShift[1][:,
                                                                                    numpy.newaxis]
        self._array = numpy.concatenate((self._array, mode))
        self.notifyObservers()

    def removeAMode(self) -> None:
        if self._array.shape[0] > 1:
            self._array = self._array[:-1, :, :]
            self.notifyObservers()
        else:
            logger.error('Refusing to remove last probe mode!')
