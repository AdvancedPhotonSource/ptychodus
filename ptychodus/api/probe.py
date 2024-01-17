from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

ProbeArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


class Probe:

    def __init__(self,
                 array: ProbeArrayType | None = None,
                 *,
                 pixelWidthInMeters: float = 0.,
                 pixelHeightInMeters: float = 0.) -> None:
        if array is None:
            self._array = numpy.zeros((1, 0, 0), dtype=complex)
        else:
            if numpy.iscomplexobj(array):
                if array.ndim == 2:
                    self._array = array[numpy.newaxis, :, :]
                elif array.ndim == 3:
                    self._array = array
                else:
                    raise ValueError('Probe must be 2- or 3-dimensional ndarray.')
            else:
                raise TypeError('Probe must be a complex-valued ndarray')

        self._pixelWidthInMeters = pixelWidthInMeters
        self._pixelHeightInMeters = pixelHeightInMeters

    @property
    def array(self) -> ProbeArrayType:
        return self._array

    @property
    def dataType(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def numberOfModes(self) -> int:
        return self._array.shape[-3]

    @property
    def heightInPixels(self) -> int:
        return self._array.shape[-2]

    @property
    def widthInPixels(self) -> int:
        return self._array.shape[-1]

    @property
    def sizeInBytes(self) -> int:
        return self._array.nbytes

    @property
    def pixelWidthInMeters(self) -> float:
        return self._pixelWidthInMeters

    @property
    def pixelHeightInMeters(self) -> float:
        return self._pixelHeightInMeters

    def getMode(self, number: int) -> ProbeArrayType:
        return self._array[number, :, :]

    def getModesFlattened(self) -> ProbeArrayType:
        if self._array.size > 0:
            return self._array.transpose((1, 0, 2)).reshape(self._array.shape[-2], -1)
        else:
            return self._array

    def getModeRelativePower(self) -> Sequence[float]:
        probe = self._array
        power = numpy.sum((probe * probe.conj()).real, axis=(-2, -1))
        powersum = power.sum()

        if powersum > 0.:
            power /= powersum

        return power


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
