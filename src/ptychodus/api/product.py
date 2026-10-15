"""Product data structure bundling the probe positions, probe sequence, object and metadata."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from sys import getsizeof

from .constants import energy_eV_to_J, energy_eV_to_wavelength_m
from .diffraction import Polarization
from .object import Object
from .probe import Probe, ProbeEnsemble
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
    probes: ProbeEnsemble
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
        """Yield ``(scan_position, probe)`` pairs for every scan position.

        Without OPR weights one probe describes the whole scan and is yielded at every
        position. With them each position takes its own weight row, and weights carrying
        more rows than the scan has positions are left unused -- a scan that is a prefix
        of the one that solved the probe still pairs row for row.

        Raises:
            ValueError: when the OPR weights describe fewer positions than the scan
                holds, which leaves later positions with no row to take. Reconcile the
                weights with an OPR weight policy before pairing them with this scan.
        """
        num_positions = len(self.probe_positions)
        num_opr_positions = self.probes.num_opr_positions

        if num_opr_positions is not None and num_opr_positions < num_positions:
            raise ValueError(
                f'Probe carries OPR weights for {num_opr_positions} probe position(s)'
                f' but the product has {num_positions}!'
            )

        for index, position in enumerate(self.probe_positions):
            yield position, self.probes.get_probe(index)


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
