from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .apparatus import ImageExtent

ProbeArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


class Probe:

    def __init__(self, array: ProbeArrayType | None = None) -> None:
        self._array = numpy.zeros((1, 0, 0), dtype=complex)

        if array is not None:
            self.setArray(array)

    def copy(self) -> Probe:
        clone = Probe()
        clone._array = self._array.copy()
        return clone

    def getArray(self) -> ProbeArrayType:
        return self._array

    def setArray(self, array: ProbeArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Probe must be a complex-valued ndarray')

        if array.ndim == 2:
            self._array = array[numpy.newaxis, :, :]
        elif array.ndim == 3:
            self._array = array
        else:
            raise ValueError('Probe must be 2- or 3-dimensional ndarray.')

    def getDataType(self) -> numpy.dtype:
        return self._array.dtype

    def getImageExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self._array.shape[-1],
            heightInPixels=self._array.shape[-2],
        )

    def getSizeInBytes(self) -> int:
        return self._array.nbytes

    def getNumberOfModes(self) -> int:
        return self._array.shape[-3]

    def getMode(self, number: int) -> ProbeArrayType:
        return self._array[number, :, :]

    def getModesFlattened(self) -> ProbeArrayType:
        if self._array.size > 0:
            return self._array.transpose((1, 0, 2)).reshape(self._array.shape[-2], -1)
        else:
            return self._array

    def getModeRelativePower(self, number: int) -> float:
        probe = self._array
        power = numpy.sum((probe * probe.conj()).real, axis=(-2, -1))
        powersum = power.sum()

        if powersum > 0.:
            power /= powersum

        return power[number]


class ProbeFileReader(ABC):

    @abstractmethod
    def read(self, filePath: Path) -> Probe:
        '''reads a probe from file'''
        pass


class ProbeFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, probe: Probe) -> None:
        '''writes a probe to file'''
        pass
