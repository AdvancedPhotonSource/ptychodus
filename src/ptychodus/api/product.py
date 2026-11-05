"""Product data structure bundling the probe positions, probe sequence, object and metadata."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from sys import getsizeof
import logging
import math

import numpy

from .affine import AffineTransform, transform_probe_positions
from .constants import energy_eV_to_J, energy_eV_to_wavelength_m
from .diffraction import Polarization
from .files import FileReader, FileWriter
from .geometry import PixelGeometry
from .interpolate import resample_complex_at
from .object import Object, ObjectGeometry, compute_object_geometry
from .probe import Probe, ProbeEnsemble, ProbeGeometry
from .probe_positions import ProbePosition, ProbePositionSequence
from .propagate import compute_magnification
from .scattering.geometry import ScatteringGeometry
from .typing import ComplexArrayType

__all__ = [
    'LossValue',
    'Product',
    'ProductFileReader',
    'ProductFileWriter',
    'ProductMetadata',
    'reconcile_defocus_distance_m',
    'transform_product',
]

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
    """Sample tilt relative to the beam in an XMCD projection, as APS 4-ID POLAR records it.

    It sets which magnetization component the XMCD contrast is sensitive to. ptychodus
    stores and persists it, but no computation reads it. It belongs to a transmission
    measurement and is unrelated to `scattering_geometry`: it neither moves the detector
    nor changes the frame the probe positions are in. A Bragg or Laue sample's orientation
    is ``scattering_geometry.incidence_angle_deg``.
    """
    polarization: Polarization | None = None
    far_field: bool = True
    """Whether the detector records the Fraunhofer diffraction pattern.

    Independent of :attr:`magnification`: a focusing optic says where the beam waist
    sits, not which propagation regime the detector samples. Far field with an optic
    and near field without one are both expressible, and both occur.
    """
    i0_photons_per_count: float = 0.0
    """Incident photons per unit of the I0 reading the probe positions carry.

    I0 -- an upstream ion chamber, diode or scaler -- is recorded in whatever units the
    instrument uses; this converts it to incident photons. Zero means uncalibrated, so
    I0 then only corrects the detector totals for beam decay.
    """
    scattering_geometry: ScatteringGeometry = ScatteringGeometry()
    """Directions of the exit beam and the sample surface; forward for transmission.

    The probe positions, probe and object of a product are all in the exit frame of this
    geometry -- the frame the detector's Fourier relation holds in, which is the ordinary
    transmission frame when the geometry is forward. Lab-frame data is carried into it
    once, where it enters a product (see
    :func:`ptychodus.api.scattering.frame.resolve_exit_frame_transform`), and never again.
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
        sz += getsizeof(self.i0_photons_per_count)
        sz += getsizeof(self.exposure_time_s)
        sz += getsizeof(self.mass_attenuation_m2_per_kg)
        sz += getsizeof(self.tomography_angle_deg)
        sz += getsizeof(self.focus_object_distance_m)
        sz += getsizeof(self.tilt_angle_deg)
        sz += getsizeof(self.polarization)
        sz += getsizeof(self.far_field)
        sz += getsizeof(self.scattering_geometry)
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

        See :meth:`ptychodus.api.probe.ProbeEnsemble.iter_position_probes`.
        """
        return self.probes.iter_position_probes(self.probe_positions)


class ProductFileReader(FileReader[Product]):
    """Plugin interface for reading data products."""


class ProductFileWriter(FileWriter[Product]):
    """Plugin interface for writing data products."""


def _calibrate_detector_distance(
    detector_distance_m: float,
    scale: float,
    *,
    focus_object_distance_m: float = 0.0,
    far_field: bool = True,
) -> float:
    """Return the detector distance implied by a fitted isotropic position scale.

    In far-field geometry the object pixel is ``wavelength * distance / (width * detector pixel)``,
    so it is proportional to the detector distance -- this inverts
    :func:`ptychodus.api.propagate.compute_far_field_pixel_geometry` with respect to the distance.
    A fit that had to spread the probe positions out by ``scale`` means the assumed object pixel
    was too large by that same factor, and the distance that reproduces the correct pixel is
    ``detector_distance_m / scale``.

    Both near-field geometries break that proportionality, for different reasons, so each is
    refused rather than calibrated. The regime has to be tested as well as the focus position:
    a parallel beam declares no focusing optic, so testing the focus alone would admit it.

    Args:
        detector_distance_m: Sample-to-detector distance the positions were reconstructed with.
        scale: Isotropic scale from :meth:`AffineTransform.decompose`.
        focus_object_distance_m: Focus-to-object distance of the product. A nonzero value selects
            cone-beam geometry, where the object pixel is set by
            :func:`ptychodus.api.propagate.compute_magnification` rather than by the detector
            distance alone and this relation does not hold.
        far_field: Declared propagation regime. Near field back-projects the detector pixel
            onto the object, so the object pixel does not depend on the detector distance at
            all and no position scale can report one.

    Raises:
        ValueError: If ``scale`` is not positive and finite, if ``detector_distance_m`` is not
            finite, if ``focus_object_distance_m`` is nonzero, or if the product declares
            near-field propagation.
    """
    if focus_object_distance_m != 0.0:
        raise ValueError(
            'Detector-distance calibration from a position scale is defined for far-field '
            'geometry only; this product has a nonzero focus-object distance '
            f'({focus_object_distance_m} m), where the object pixel follows the magnification.'
        )

    if not far_field:
        raise ValueError(
            'Detector-distance calibration from a position scale is defined for far-field '
            'geometry only; this product declares near-field propagation, where the object '
            'pixel is the detector pixel back-projected and carries no distance to recover.'
        )

    if not math.isfinite(detector_distance_m):
        raise ValueError(f'Detector distance must be finite; got {detector_distance_m}.')

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'Position scale must be positive and finite; got {scale}.')

    return detector_distance_m / scale


def _target_object_geometry(
    source_geometry: ObjectGeometry,
    positions: ProbePositionSequence,
    corrected_positions: ProbePositionSequence,
    probe_geometry: ProbeGeometry,
    corrected_probe_geometry: ProbeGeometry,
    scale_free: AffineTransform,
) -> ObjectGeometry:
    """Return the grid the corrected object lives on.

    How much the canvas changes is decided by :func:`compute_object_geometry`, not here: it is
    evaluated once on the uncorrected scan and probe and once on the corrected ones, and the
    difference between the two is the change. Whatever the source object has beyond that canonical
    size -- extra padding, or simply the size a warm start read from file happens to be -- rides
    through as a signed per-axis difference, so an identity transform is an exact no-op and no
    object is silently cropped to the minimal canvas.
    """
    base_source = compute_object_geometry(positions, probe_geometry)
    base_target = compute_object_geometry(corrected_positions, corrected_probe_geometry)
    extra_width_px = source_geometry.width_px - base_source.width_px
    extra_height_px = source_geometry.height_px - base_source.height_px
    width_px = base_target.width_px + extra_width_px
    height_px = base_target.height_px + extra_height_px

    if width_px < 1 or height_px < 1:
        raise ValueError(
            f'The corrected object canvas is {width_px} x {height_px} px; the source object is '
            f'{-extra_width_px} x {-extra_height_px} px smaller than the canvas its own scan '
            'needs, and the correction shrinks it past nothing.'
        )

    center = scale_free(ProbePosition(0, source_geometry.center_x_m, source_geometry.center_y_m))
    return ObjectGeometry(
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=base_target.pixel_width_m,
        pixel_height_m=base_target.pixel_height_m,
        center_x_m=center.x_m,
        center_y_m=center.y_m,
    )


def _resample_object(
    array: ComplexArrayType,
    source_geometry: ObjectGeometry,
    target_geometry: ObjectGeometry,
    transform: AffineTransform,
    *,
    interpolation_order: int,
) -> ComplexArrayType:
    """Resample every layer of *array* from its own grid onto the target grid.

    Source and target differ in extent, pitch and center all at once, so the source index of each
    target pixel is built explicitly: take the target pixel's physical coordinate, map it back
    through the inverse of *transform*, and express the result in the source's fractional pixel
    indexes under the centered-pixel convention. Target pixels falling outside the source array
    are filled with zeros, which is what padding an object does elsewhere.
    """
    coordinates = target_geometry.get_transverse_coordinates()
    x_m = target_geometry.center_x_m + coordinates.x_m
    y_m = target_geometry.center_y_m + coordinates.y_m

    source_x_m, source_y_m = transform.inverted().transform_coordinates(x_m, y_m)

    # Centered-pixel convention: the world center sits at pixel index (N - 1) / 2.
    column = (source_x_m - source_geometry.center_x_m) / source_geometry.pixel_width_m + (
        source_geometry.width_px - 1
    ) / 2
    row = (source_y_m - source_geometry.center_y_m) / source_geometry.pixel_height_m + (
        source_geometry.height_px - 1
    ) / 2

    resampled = numpy.empty(
        (array.shape[0], target_geometry.height_px, target_geometry.width_px), dtype=array.dtype
    )

    for index, layer in enumerate(array):
        # 'constant' rather than the helper's 'grid-constant': a target pixel past the source
        # edge is zero outright rather than a fade toward it.
        resampled[index] = resample_complex_at(
            layer, row, column, order=interpolation_order, mode='constant'
        )

    return resampled


def transform_product(
    product: Product, transform: AffineTransform, *, interpolation_order: int = 1
) -> Product:
    """Apply an affine transform to a product by splitting it between positions and geometry.

    An isotropic scale on the probe positions and the object pixel size are the same degree of
    freedom expressed two ways, so applying the whole transform to the positions *and* correcting
    the detector distance would count the scale twice. This decomposes the transform and folds
    only ``scale`` into ``detector_distance_m``, treating the scan stage as the trustworthy
    measurement and the detector distance as the miscalibrated one. Because the far-field object
    pixel is proportional to that distance, the same scale divides both pixel geometries exactly,
    with no interpolation. The remaining scale-free part -- asymmetry, rotation and shear -- moves
    the probe positions and warps the object array onto the grid they now imply.

    The object's array therefore changes shape as well as pitch, and the two axes change
    independently: a finer pixel needs more of them to cover the same scan, and asymmetry and
    rotation act differently on each axis. The probe array is reinterpreted rather than resampled,
    since the scale-free part distorts the scan frame and not the illumination optics.

    The correction is translation-free. A translation shifts the positions and the object center
    by the same amount, so it relabels the product's lab coordinates without changing a single
    sample; relocating products against each other is a cross-scan registration job.

    Resampling costs accuracy. Measured relative error of one bilinear pass against a band-limited
    reference, as a function of how far the object's content reaches toward Nyquist: 1.3e-02 at
    20%, 1.2e-01 at 40%, 4.3e-01 at 70%. Raising ``interpolation_order`` to 3 buys roughly an
    order of magnitude on smooth content at three times the cost, and buys much less on content
    that reaches Nyquist. The warp is small in absolute terms -- a corner moves 0.43 px on a 256 px
    canvas and 1.72 px on a 1024 px one at typical fitted magnitudes -- and the dominant
    component, the isotropic scale, never reaches the interpolator at all.

    Args:
        product: Product to correct.
        transform: Affine map fitted to this product's probe positions.
        interpolation_order: Spline order passed to :func:`scipy.ndimage.map_coordinates` for the
            object array. Raising it sharpens the resampled object at superlinear cost.

    Raises:
        ValueError: If the linear part is singular or orientation-reversing, if the product is not
            far-field, or if the correction shrinks the object canvas away entirely.
    """
    metadata = product.metadata
    components = transform.decompose()
    detector_distance_m = _calibrate_detector_distance(
        metadata.detector_distance_m,
        components.scale,
        focus_object_distance_m=metadata.focus_object_distance_m,
        far_field=metadata.far_field,
    )
    scale_free = AffineTransform.from_components(replace(components, scale=1.0))

    probes = product.probes
    probe_geometry = probes.get_geometry()
    corrected_probe_pixel_geometry = PixelGeometry(
        width_m=probe_geometry.pixel_width_m / components.scale,
        height_m=probe_geometry.pixel_height_m / components.scale,
    )
    corrected_probe_geometry = ProbeGeometry(
        width_px=probe_geometry.width_px,
        height_px=probe_geometry.height_px,
        pixel_width_m=corrected_probe_pixel_geometry.width_m,
        pixel_height_m=corrected_probe_pixel_geometry.height_m,
    )

    positions = product.probe_positions
    corrected_positions = ProbePositionSequence(
        list(transform_probe_positions(positions, scale_free))
    )

    object_ = product.object_
    source_geometry = object_.get_geometry()
    target_geometry = _target_object_geometry(
        source_geometry,
        positions,
        corrected_positions,
        probe_geometry,
        corrected_probe_geometry,
        scale_free,
    )
    logger.info(
        'Scale %.6f moves the detector distance from %.6f m to %.6f m; applying asymmetry '
        '%+.3e, rotation %+.4f deg and shear %+.4f deg to %d probe positions and resampling the '
        'object from %s onto %s.',
        components.scale,
        metadata.detector_distance_m,
        detector_distance_m,
        components.asymmetry,
        math.degrees(components.rotation_rad),
        math.degrees(components.shear_rad),
        len(positions),
        source_geometry,
        target_geometry,
    )
    return replace(
        product,
        metadata=replace(metadata, detector_distance_m=detector_distance_m),
        probe_positions=corrected_positions,
        probes=ProbeEnsemble(
            array=probes.get_array(),
            opr_weights=probes.get_opr_weights_or_none(),
            pixel_geometry=corrected_probe_pixel_geometry,
        ),
        object_=Object(
            array=_resample_object(
                object_.get_array(),
                source_geometry,
                target_geometry,
                scale_free,
                interpolation_order=interpolation_order,
            ),
            pixel_geometry=target_geometry.get_pixel_geometry(),
            center=target_geometry.get_center(),
            layer_spacing_m=list(object_.layer_spacing_m),
        ),
    )
