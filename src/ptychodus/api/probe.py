from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy

from .geometry import PixelGeometry
from .propagator import intensity
from .typing import ComplexArrayType, RealArrayType


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

    @property
    @abstractmethod
    def num_scan_points(self) -> int:
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
        array: ComplexArrayType,
        pixel_geometry: PixelGeometry,
    ) -> None:
        self._array = array
        self._pixel_geometry = pixel_geometry

        if array.ndim != 3:
            raise ValueError('Probe must be a 3-dimensional ndarray.')

        power = numpy.sum(intensity(array), axis=(-2, -1))
        powersum = numpy.sum(power)

        if powersum > 0.0:
            power /= powersum

        self._mode_relative_power = power.tolist()

    def copy(self) -> Probe:
        return Probe(
            array=self._array.copy(),
            pixel_geometry=self._pixel_geometry.copy(),
        )

    def get_array(self) -> ComplexArrayType:
        return self._array

    def get_pixel_geometry(self) -> PixelGeometry:
        return self._pixel_geometry

    @property
    def dtype(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def width_px(self) -> int:
        return self._array.shape[-1]

    @property
    def height_px(self) -> int:
        return self._array.shape[-2]

    @property
    def num_incoherent_modes(self) -> int:
        return self._array.shape[-3]

    def get_incoherent_mode(self, number: int) -> ComplexArrayType:
        return self._array[number, :, :]

    def get_incoherent_modes_flattened(self) -> ComplexArrayType:
        return self._array.transpose((1, 0, 2)).reshape(self.height_px, -1)

    def get_incoherent_mode_relative_power(self, number: int) -> float:
        return self._mode_relative_power[number]

    def get_coherence(self) -> float:
        return numpy.sqrt(numpy.sum(numpy.square(self._mode_relative_power)))

    def get_intensity(self) -> RealArrayType:
        return numpy.sum(intensity(self._array), axis=-3)


class ProbeSequence(Sequence[Probe]):
    def __init__(
        self,
        array: ComplexArrayType | None,
        opr_weights: RealArrayType | None,
        pixel_geometry: PixelGeometry | None,
    ) -> None:
        if array is None:
            self._array: ComplexArrayType = numpy.zeros((1, 1, 0, 0), dtype=complex)
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

        if opr_weights is None:
            self._opr_weights = None
        elif numpy.issubdtype(opr_weights.dtype, numpy.floating):
            if opr_weights.ndim == 2:
                num_weights_actual = opr_weights.shape[1]
                num_weights_expected = self._array.shape[0]

                if num_weights_actual == num_weights_expected:
                    self._opr_weights = opr_weights
                else:
                    raise ValueError(
                        (
                            'inconsistent number of opr weights!'
                            f' actual={num_weights_actual}'
                            f' expected={num_weights_expected}'
                        )
                    )
            else:
                raise ValueError('opr_weights must be 2-dimensional ndarray')
        else:
            raise TypeError('opr_weights must be a floating-point ndarray')

        self._pixel_geometry = pixel_geometry

    def copy(self) -> ProbeSequence:
        return ProbeSequence(
            self._array.copy(),
            None if self._opr_weights is None else self._opr_weights.copy(),
            None if self._pixel_geometry is None else self._pixel_geometry.copy(),
        )

    def get_array(self) -> ComplexArrayType:
        return self._array

    def get_opr_weights(self) -> RealArrayType:
        if self._opr_weights is None:
            raise ValueError('Missing opr_weights!')

        return self._opr_weights

    def get_pixel_geometry(self) -> PixelGeometry:
        if self._pixel_geometry is None:
            raise ValueError('Missing probe pixel geometry!')

        return self._pixel_geometry

    @property
    def dtype(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def nbytes(self) -> int:
        sz = self._array.nbytes

        if self._opr_weights is not None:
            sz += self._opr_weights.nbytes

        return sz

    @property
    def num_coherent_modes(self) -> int:
        return self._array.shape[0]

    @property
    def num_incoherent_modes(self) -> int:
        return self._array.shape[1]

    @property
    def height_px(self) -> int:
        return self._array.shape[2]

    @property
    def width_px(self) -> int:
        return self._array.shape[3]

    @overload
    def __getitem__(self, index: int) -> Probe: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Probe]: ...

    def __getitem__(self, index: int | slice) -> Probe | Sequence[Probe]:
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start, index.stop, index.step)]

        array = self._array[0, :, :, :].copy()

        if self._opr_weights is not None:
            array[0, :, :] = numpy.tensordot(
                self._opr_weights[index, :], self._array[:, 0, :, :], axes=1
            )

        return Probe(array, self.get_pixel_geometry())

    def get_probe_no_opr(self) -> Probe:
        array = self._array[0, :, :, :].copy()
        return Probe(array, self.get_pixel_geometry())

    def get_geometry(self) -> ProbeGeometry:
        pixel_geometry = self.get_pixel_geometry()

        return ProbeGeometry(
            width_px=self.width_px,
            height_px=self.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )

    def __len__(self) -> int:
        return 1 if self._opr_weights is None else self._opr_weights.shape[0]

    def __repr__(self) -> str:
        return f'{self._array.dtype}{self._array.shape}'


class ProbeFileReader(ABC):
    @abstractmethod
    def read(self, file_path: Path) -> ProbeSequence:
        """reads a probe from file"""
        pass


class ProbeFileWriter(ABC):
    @abstractmethod
    def write(self, file_path: Path, probes: ProbeSequence) -> None:
        """writes a probe to file"""
        pass
