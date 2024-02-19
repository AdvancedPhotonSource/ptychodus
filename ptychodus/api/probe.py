from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .patterns import ImageExtent

ProbeArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


@dataclass(frozen=True)
class ProbeGeometry:
    widthInPixels: int
    heightInPixels: int
    pixelWidthInMeters: float
    pixelHeightInMeters: float

    @property
    def widthInMeters(self) -> float:
        return self.widthInPixels * self.pixelWidthInMeters

    @property
    def heightInMeters(self) -> float:
        return self.heightInPixels * self.pixelHeightInMeters

    def _asTuple(self) -> tuple[int, int, float, float]:
        return (self.widthInPixels, self.heightInPixels, self.pixelWidthInMeters,
                self.pixelHeightInMeters)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ProbeGeometry):
            return (self._asTuple() == other._asTuple())

        return False


class ProbeGeometryProvider(ABC):

    @property
    @abstractmethod
    def probeWavelengthInMeters(self) -> float:
        pass

    @abstractmethod
    def getProbeGeometry(self) -> ProbeGeometry:
        pass


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

    def copy(self) -> Probe:
        return Probe(
            array=numpy.array(self._array),
            pixelWidthInMeters=float(self._pixelWidthInMeters),
            pixelHeightInMeters=float(self._pixelHeightInMeters),
        )

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
    def sizeInBytes(self) -> int:
        return self._array.nbytes

    @property
    def pixelWidthInMeters(self) -> float:
        return self._pixelWidthInMeters

    @property
    def pixelHeightInMeters(self) -> float:
        return self._pixelHeightInMeters

    def getGeometry(self) -> ProbeGeometry:
        return ProbeGeometry(
            widthInPixels=self._array.shape[-1],
            heightInPixels=self._array.shape[-2],
            pixelWidthInMeters=self._pixelWidthInMeters,
            pixelHeightInMeters=self._pixelHeightInMeters,
        )

    def getExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self._array.shape[-1],
            heightInPixels=self._array.shape[-2],
        )

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
