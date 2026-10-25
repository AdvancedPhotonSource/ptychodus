"""Product data structure bundling the probe positions, probe sequence, object and metadata."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from sys import getsizeof
import logging
import math

from .constants import energy_eV_to_J, energy_eV_to_wavelength_m
from .diffraction import Polarization
from .files import FileReader, FileWriter
from .object import Object
from .probe import Probe, ProbeEnsemble
from .probe_positions import ProbePosition, ProbePositionSequence
from .propagate import compute_magnification

logger = logging.getLogger(__name__)


def reconcile_defocus_distance_m(
    focus_object_distance_m: float,
    defocus_distance_m: float,
    *,
    rel_tol: float = 1.0e-6,
) -> float:
    """How far past its focus a probe is sampled, given the two frames that state it.

    The same displacement is stated twice, in opposite frames: as a product's signed
    focus position *focus_object_distance_m*, and as a probe's own defocus
    *defocus_distance_m*. The defocus is the negated focus position, so a focus upstream
    of the object is a probe sampled downstream of it. The product wins wherever it
    names a focus, because the product is what the reconstruction geometry is computed
    from -- a probe built against a different focus than the geometry would disagree
    with the object pixel size it is reconstructed on.

    A focus position of zero means no focusing optic, which cannot describe a focused
    probe, so *defocus_distance_m* is returned unchanged. A disagreement beyond
    *rel_tol* is reported rather than silently resolved; widen it only where the two are
    expected to differ in their last digits.
    """
    if focus_object_distance_m == 0.0:
        return defocus_distance_m

    from_product_m = -focus_object_distance_m

    if defocus_distance_m != 0.0 and not math.isclose(
        defocus_distance_m, from_product_m, rel_tol=rel_tol
    ):
        logger.warning(
            'Probe defocus %g m disagrees with the focus-object distance %g m the product '
            'declares, which implies %g m; using the product.',
            defocus_distance_m,
            focus_object_distance_m,
            from_product_m,
        )

    return from_product_m


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
    def defocus_distance_m(self) -> float:
        """How far past the focus the object sits, the negated `focus_object_distance_m`.

        Zero when the product declares no focusing optic. See
        :func:`reconcile_defocus_distance_m` for combining it with a probe's own defocus.
        """
        return -self.focus_object_distance_m

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


class ProductFileReader(FileReader[Product]):
    """Plugin interface for reading data products."""


class ProductFileWriter(FileWriter[Product]):
    """Plugin interface for writing data products."""
