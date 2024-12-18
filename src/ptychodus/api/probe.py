from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy

from .geometry import PixelGeometry
from .propagator import WavefieldArrayType, intensity
from .typing import RealArrayType


@dataclass(frozen=True)
class FresnelZonePlate:
    zonePlateDiameterInMeters: float
    outermostZoneWidthInMeters: float
    centralBeamstopDiameterInMeters: float

    def getFocalLengthInMeters(self, centralWavelengthInMeters: float) -> float:
        return (
            self.zonePlateDiameterInMeters
            * self.outermostZoneWidthInMeters
            / centralWavelengthInMeters
        )


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

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self.pixelWidthInMeters,
            heightInMeters=self.pixelHeightInMeters,
        )


class ProbeGeometryProvider(ABC):
    @property
    @abstractmethod
    def detectorDistanceInMeters(self) -> float:
        pass

    @property
    @abstractmethod
    def probePhotonCount(self) -> float:
        pass

    @property
    @abstractmethod
    def probeWavelengthInMeters(self) -> float:
        pass

    @property
    @abstractmethod
    def probePowerInWatts(self) -> float:
        pass

    @abstractmethod
    def getProbeGeometry(self) -> ProbeGeometry:
        pass


class Probe:
    def __init__(
        self,
        array: WavefieldArrayType | None,
        pixelGeometry: PixelGeometry | None,
    ) -> None:
        if array is None:
            self._array = numpy.zeros((1, 1, 0, 0), dtype=complex)
        elif numpy.iscomplexobj(array):
            match array.ndim:
                case 2:
                    self._array = array[numpy.newaxis, numpy.newaxis, ...]
                case 3:
                    self._array = array[numpy.newaxis, ...]
                case 4:
                    self._array = array
                case _:
                    raise ValueError('Probe must be 2-, 3-, or 4-dimensional ndarray.')
        else:
            raise TypeError('Probe must be a complex-valued ndarray')

        self._pixelGeometry = pixelGeometry

        power = numpy.sum(intensity(self._array[0]), axis=(-2, -1))
        powersum = numpy.sum(power)

        if powersum > 0.0:
            power /= powersum

        self._modeRelativePower = power.tolist()

    def copy(self) -> Probe:
        return Probe(
            array=self._array.copy(),
            pixelGeometry=None if self._pixelGeometry is None else self._pixelGeometry.copy(),
        )

    def getArray(self) -> WavefieldArrayType:
        return self._array

    @property
    def dataType(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def sizeInBytes(self) -> int:
        return self._array.nbytes

    @property
    def widthInPixels(self) -> int:
        return self._array.shape[-1]

    @property
    def heightInPixels(self) -> int:
        return self._array.shape[-2]

    @property
    def numberOfIncoherentModes(self) -> int:
        return self._array.shape[-3]

    @property
    def numberOfCoherentModes(self) -> int:
        return self._array.shape[-4]

    def getPixelGeometry(self) -> PixelGeometry | None:
        return self._pixelGeometry

    def getGeometry(self) -> ProbeGeometry:
        pixelWidthInMeters = 0.0
        pixelHeightInMeters = 0.0

        if self._pixelGeometry is not None:
            pixelWidthInMeters = self._pixelGeometry.widthInMeters
            pixelHeightInMeters = self._pixelGeometry.heightInMeters

        return ProbeGeometry(
            widthInPixels=self.widthInPixels,
            heightInPixels=self.heightInPixels,
            pixelWidthInMeters=pixelWidthInMeters,
            pixelHeightInMeters=pixelHeightInMeters,
        )

    def getIncoherentMode(self, number: int) -> WavefieldArrayType:
        return self._array[0, number, :, :]

    def getIncoherentModesFlattened(self) -> WavefieldArrayType:
        modes = self._array[0]
        return modes.transpose((1, 0, 2)).reshape(modes.shape[-2], -1)

    def getIncoherentModeRelativePower(self, number: int) -> float:
        return self._modeRelativePower[number]

    def getCoherence(self) -> float:
        return numpy.sqrt(numpy.sum(numpy.square(self._modeRelativePower)))

    def getCoherentMode(self, number: int) -> WavefieldArrayType:
        return self._array[number, 0, :, :]

    def getIntensity(self) -> RealArrayType:
        array_no_opr = self._array[0]  # TODO OPR
        return numpy.sum(intensity(array_no_opr), axis=-3)


class ProbeFileReader(ABC):
    @abstractmethod
    def read(self, filePath: Path) -> Probe:
        """reads a probe from file"""
        pass


class ProbeFileWriter(ABC):
    @abstractmethod
    def write(self, filePath: Path, probe: Probe) -> None:
        """writes a probe to file"""
        pass
