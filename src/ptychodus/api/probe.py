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
    zone_plate_diameter_m: float
    outermost_zone_width_m: float
    central_beamstop_diameter_m: float

    def get_focal_length_m(self, central_wavelength_m: float) -> float:
        return self.zone_plate_diameter_m * self.outermost_zone_width_m / central_wavelength_m


@dataclass(frozen=True)
class ProbeGeometry:
    width_px: int
    height_px: int
    pixel_width_m: float
    pixel_height_m: float

    @property
    def width_m(self) -> float:
        return self.width_px * self.pixel_width_m

    @property
    def height_m(self) -> float:
        return self.height_px * self.pixel_height_m

    def get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self.pixel_width_m,
            height_m=self.pixel_height_m,
        )


class ProbeGeometryProvider(ABC):
    @property
    @abstractmethod
    def detector_distance_m(self) -> float:
        pass

    @property
    @abstractmethod
    def probe_photon_count(self) -> float:
        pass

    @property
    @abstractmethod
    def probe_wavelength_m(self) -> float:
        pass

    @property
    @abstractmethod
    def probe_power_W(self) -> float:  # noqa: N802
        pass

    @abstractmethod
    def get_detector_pixel_geometry(self) -> PixelGeometry:
        pass

    @abstractmethod
    def get_probe_geometry(self) -> ProbeGeometry:
        pass


class Probe:
    def __init__(
        self,
        array: WavefieldArrayType | None,
        pixel_geometry: PixelGeometry | None,
    ) -> None:
        if array is None:
            self._array: WavefieldArrayType = numpy.zeros((1, 1, 0, 0), dtype=complex)
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

        self._pixel_geometry = pixel_geometry

        power = numpy.sum(intensity(self._array[0]), axis=(-2, -1))
        powersum = numpy.sum(power)

        if powersum > 0.0:
            power /= powersum

        self._mode_relative_power = power.tolist()

    def copy(self) -> Probe:
        return Probe(
            array=self._array.copy(),
            pixel_geometry=None if self._pixel_geometry is None else self._pixel_geometry.copy(),
        )

    def get_array(self) -> WavefieldArrayType:
        return self._array

    @property
    def dtype(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    @property
    def width_px(self) -> int:
        return self._array.shape[-1]

    @property
    def height_px(self) -> int:
        return self._array.shape[-2]

    @property
    def num_incoherent_modes(self) -> int:
        return self._array.shape[-3]

    @property
    def num_coherent_modes(self) -> int:
        return self._array.shape[-4]

    def get_pixel_geometry(self) -> PixelGeometry | None:
        return self._pixel_geometry

    def get_geometry(self) -> ProbeGeometry:
        pixel_width_m = 0.0
        pixel_height_m = 0.0

        if self._pixel_geometry is not None:
            pixel_width_m = self._pixel_geometry.width_m
            pixel_height_m = self._pixel_geometry.height_m

        return ProbeGeometry(
            width_px=self.width_px,
            height_px=self.height_px,
            pixel_width_m=pixel_width_m,
            pixel_height_m=pixel_height_m,
        )

    def get_incoherent_mode(self, number: int) -> WavefieldArrayType:
        return self._array[0, number, :, :]

    def get_incoherent_modes_flattened(self) -> WavefieldArrayType:
        modes = self._array[0]
        return modes.transpose((1, 0, 2)).reshape(modes.shape[-2], -1)

    def get_incoherent_mode_relative_power(self, number: int) -> float:
        return self._mode_relative_power[number]

    def get_coherence(self) -> float:
        return numpy.sqrt(numpy.sum(numpy.square(self._mode_relative_power)))

    def get_coherent_mode(self, number: int) -> WavefieldArrayType:
        return self._array[number, 0, :, :]

    def get_intensity(self) -> RealArrayType:
        array_no_opr = self._array[0]  # TODO OPR
        return numpy.sum(intensity(array_no_opr), axis=-3)


class ProbeFileReader(ABC):
    @abstractmethod
    def read(self, file_path: Path) -> Probe:
        """reads a probe from file"""
        pass


class ProbeFileWriter(ABC):
    @abstractmethod
    def write(self, file_path: Path, probe: Probe) -> None:
        """writes a probe to file"""
        pass
