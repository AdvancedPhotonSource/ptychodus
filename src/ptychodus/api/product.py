"""Product data structure bundling the probe positions, probe sequence, object and metadata."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from sys import getsizeof

from .constants import ELECTRON_VOLT_J, energy_eV_to_wavelength_m
from .diffraction import Polarization
from .object import Object
from .probe import Probe, ProbeSequence
from .probe_positions import ProbePosition, ProbePositionSequence


@dataclass(frozen=True)
class ProductMetadata:
    """Metadata for the sample and experiment geometry."""

    name: str
    comments: str
    detector_distance_m: float
    probe_energy_eV: float  # noqa: N815
    probe_photon_count: float
    exposure_time_s: float
    mass_attenuation_m2_kg: float
    tomography_angle_deg: float
    tilt_angle_deg: float = 0.0
    polarization: Polarization | None = None

    @property
    def probe_energy_J(self) -> float:  # noqa: N802
        return self.probe_energy_eV * ELECTRON_VOLT_J

    @property
    def probe_wavelength_m(self) -> float:
        return energy_eV_to_wavelength_m(self.probe_energy_eV)

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
        sz += getsizeof(self.tilt_angle_deg)
        sz += getsizeof(self.polarization)
        return sz


@dataclass(frozen=True)
class LossValue:
    """Loss recorded at a given epoch."""

    epoch: int
    value: float


@dataclass(frozen=True)
class Product:
    """A Data Product bundles metadata, positions, probes, object, and loss history."""

    metadata: ProductMetadata
    probe_positions: ProbePositionSequence
    probes: ProbeSequence
    object_: Object
    losses: Sequence[LossValue]

    @property
    def nbytes(self) -> int:
        sz = self.metadata.nbytes
        sz += self.probe_positions.nbytes
        sz += self.probes.nbytes
        sz += self.object_.nbytes
        return sz

    def iter_position_probes(self) -> Iterator[tuple[ProbePosition, Probe]]:
        """Yield ``(scan_position, probe)`` pairs for every scan position."""
        for index, position in enumerate(self.probe_positions):
            yield position, self.probes[index]


class ProductFileReader(ABC):
    """Plugin interface for reading data products."""

    @abstractmethod
    def read(self, file_path: Path) -> Product:
        """Read a data product from file."""
        pass


class ProductFileWriter(ABC):
    """Plugin interface for writing data products."""

    @abstractmethod
    def write(self, file_path: Path, product: Product) -> None:
        """Write a data product to file."""
        pass
