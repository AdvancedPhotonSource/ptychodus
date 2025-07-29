from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Final
from dataclasses import dataclass
from pathlib import Path
from sys import getsizeof

from .object import Object
from .probe import ProbeSequence
from .scan import PositionSequence

# Source: https://physics.nist.gov/cuu/Constants/index.html
ELECTRON_VOLT_J: Final[float] = 1.602176634e-19
LIGHT_SPEED_M_PER_S: Final[float] = 299792458
PLANCK_CONSTANT_J_PER_HZ: Final[float] = 6.62607015e-34


@dataclass(frozen=True)
class ProductMetadata:
    name: str
    comments: str
    detector_distance_m: float
    probe_energy_eV: float  # noqa: N815
    probe_photon_count: float
    exposure_time_s: float
    mass_attenuation_m2_kg: float
    tomography_angle_deg: float

    @property
    def probe_energy_J(self) -> float:  # noqa: N802
        return self.probe_energy_eV * ELECTRON_VOLT_J

    @property
    def probe_wavelength_m(self) -> float:
        hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S  # noqa: N806

        try:
            return hc_Jm / self.probe_energy_J
        except ZeroDivisionError:
            return 0.0

    @property
    def nbytes(self) -> int:
        sz = getsizeof(self.name)
        sz += getsizeof(self.comments)
        sz += getsizeof(self.detector_distance_m)
        sz += getsizeof(self.probe_energy_eV)
        sz += getsizeof(self.probe_photon_count)
        sz += getsizeof(self.exposure_time_s)
        sz += getsizeof(self.mass_attenuation_m2_kg)
        sz += getsizeof(self.tomography_angle_deg)
        return sz


@dataclass(frozen=True)
class LossValue:
    epoch: int
    value: float


@dataclass(frozen=True)
class Product:
    metadata: ProductMetadata
    positions: PositionSequence
    probes: ProbeSequence
    object_: Object
    losses: Sequence[LossValue]

    @property
    def nbytes(self) -> int:
        sz = self.metadata.nbytes
        sz += self.positions.nbytes
        sz += self.probes.nbytes
        sz += self.object_.nbytes
        return sz


class ProductFileReader(ABC):
    @abstractmethod
    def read(self, file_path: Path) -> Product:
        """reads a product from file"""
        pass


class ProductFileWriter(ABC):
    @abstractmethod
    def write(self, file_path: Path, product: Product) -> None:
        """writes a product to file"""
        pass
