"""Product data structure bundling the probe positions, probe sequence, object and metadata."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from sys import getsizeof

from .constants import energy_eV_to_J, energy_eV_to_wavelength_m
from .diffraction import Polarization
from .object import Object
from .probe import Probe, ProbeSequence
from .probe_positions import ProbePosition, ProbePositionSequence
from .propagate import compute_magnification


@dataclass(frozen=True)
class ProductMetadata:
    """Metadata for the sample and experiment geometry."""

    name: str
    comments: str
    detector_distance_m: float
    photon_energy_eV: float  # noqa: N815
    probe_photon_count: float
    exposure_time_s: float
    mass_attenuation_m2_per_kg: float
    tomography_angle_deg: float
    focus_object_distance_m: float = 0.0
    tilt_angle_deg: float = 0.0
    polarization: Polarization | None = None
    far_field: bool = True
    """Whether the detector records the Fraunhofer diffraction pattern.

    Independent of :attr:`magnification`: a focusing optic says where the beam waist
    sits, not which propagation regime the detector samples. Far field with an optic
    and near field without one are both expressible, and both occur.
    """

    @property
    def magnification(self) -> float:
        """See :func:`ptychodus.api.propagate.compute_magnification`."""
        return compute_magnification(self.detector_distance_m, self.focus_object_distance_m)

    @property
    def photon_energy_J(self) -> float:  # noqa: N802
        return energy_eV_to_J(self.photon_energy_eV)

    @property
    def photon_wavelength_m(self) -> float:
        return energy_eV_to_wavelength_m(self.photon_energy_eV)

    @property
    def nbytes(self) -> int:
        sz = getsizeof(self.name)
        sz += getsizeof(self.comments)
        sz += getsizeof(self.detector_distance_m)
        sz += getsizeof(self.photon_energy_eV)
        sz += getsizeof(self.probe_photon_count)
        sz += getsizeof(self.exposure_time_s)
        sz += getsizeof(self.mass_attenuation_m2_per_kg)
        sz += getsizeof(self.tomography_angle_deg)
        sz += getsizeof(self.focus_object_distance_m)
        sz += getsizeof(self.tilt_angle_deg)
        sz += getsizeof(self.polarization)
        sz += getsizeof(self.far_field)
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
