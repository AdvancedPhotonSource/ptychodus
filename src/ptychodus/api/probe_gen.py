"""Probe generation functions: geometric apertures, zone plates, Zernike modes, Hermite modes, and OPR ensembles."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import logging

import numpy
import scipy.linalg

from .common import ComplexArrayType, RealArrayType
from .geometry import HermiteMode, PixelGeometry, ZernikeMode
from .probe import Probe, ProbeGeometry, ProbeSequence
from .propagator import (
    AngularSpectrumPropagator,
    FresnelTransformPropagator,
    PropagatorParameters,
    intensity,
)
from .reconstructor import AssembledDiffractionData


logger = logging.getLogger(__name__)


def rescale_probe_intensity(probe: Probe, new_intensity: float) -> Probe:
    """Return a copy of *probe* rescaled so its total intensity equals *new_intensity*."""
    array = probe.get_array()
    old_intensity = numpy.sum(intensity(array))

    if new_intensity <= 0:
        logger.warning('Refusing to rescale probe to zero intensity!')
    elif numpy.isnan(old_intensity):
        logger.warning('Cannot rescale probe with NaN values!')
    elif old_intensity <= 0:
        logger.warning('Cannot rescale probe with zero intensity!')
    else:
        return Probe(
            array=array * numpy.sqrt(new_intensity / old_intensity),
            pixel_geometry=probe.get_pixel_geometry(),
        )

    return probe


def defocus_probe(
    probe: Probe,
    *,
    probe_wavelength_m: float,
    defocus_distance_m: float,
) -> Probe:
    """Propagate a probe by *defocus_distance_m* using the angular-spectrum method."""
    pixel_geometry = probe.get_pixel_geometry()
    propagator_parameters = PropagatorParameters(
        wavelength_m=probe_wavelength_m,
        width_px=probe.width_px,
        height_px=probe.height_px,
        pixel_width_m=pixel_geometry.width_m,
        pixel_height_m=pixel_geometry.height_m,
        propagation_distance_m=defocus_distance_m,
    )
    propagator = AngularSpectrumPropagator(propagator_parameters)
    return Probe(
        array=propagator.propagate(probe.get_array()),
        pixel_geometry=pixel_geometry,
    )


def generate_disk_probe(
    geometry: ProbeGeometry,
    *,
    radius_m: float,
) -> Probe:
    """Generate a binary circular aperture probe with the given radius."""
    coords = geometry.get_transverse_coordinates()
    return Probe(
        array=numpy.where(coords.position_r_m < radius_m, 1, 0) + 0j,
        pixel_geometry=geometry.get_pixel_geometry(),
    )


def generate_rectangular_probe(
    geometry: ProbeGeometry,
    *,
    width_m: float,
    height_m: float,
) -> Probe:
    """Generate a binary rectangular aperture probe with the given physical dimensions."""
    coords = geometry.get_transverse_coordinates()
    is_inside = numpy.logical_and(
        numpy.fabs(coords.position_x_m) < 0.5 * width_m,
        numpy.fabs(coords.position_y_m) < 0.5 * height_m,
    )
    return Probe(
        array=numpy.where(is_inside, 1, 0) + 0j,
        pixel_geometry=geometry.get_pixel_geometry(),
    )


def generate_super_gaussian_probe(
    geometry: ProbeGeometry,
    *,
    annular_radius_m: float,
    fwhm_m: float,
    order_parameter: float,
) -> Probe:
    """Generate a super-Gaussian (possibly annular) probe with tunable ring radius and order."""
    coords = geometry.get_transverse_coordinates()
    z = (coords.position_r_m - annular_radius_m) / fwhm_m
    zp = numpy.power(2 * z, 2 * order_parameter)
    return Probe(
        array=numpy.exp(-numpy.log(2) * zp) + 0j,
        pixel_geometry=geometry.get_pixel_geometry(),
    )


def generate_average_pattern_probe(
    geometry: ProbeGeometry,
    assembled_data: AssembledDiffractionData,
    *,
    probe_wavelength_m: float,
    detector_distance_m: float,
    rtol: float = 1.0e-3,
) -> Probe:
    """Back-propagate the square root of the mean diffraction pattern to estimate the probe.

    Raises ValueError if the diffraction-pattern shape or the Fresnel-transform's implied
    sample-plane pixel size are inconsistent with *geometry*. The single-FFT Fresnel
    propagator preserves array shape, and its output pitch is ``lambda * |z| / (N * dx_det)``;
    both must match what *geometry* declares for the returned Probe to be self-consistent.
    """
    detector_intensity = numpy.mean(assembled_data.get_patterns(), axis=0)
    height_px, width_px = detector_intensity.shape[-2:]

    if (width_px, height_px) != (geometry.width_px, geometry.height_px):
        raise ValueError(
            f'Diffraction pattern shape ({width_px}x{height_px} px) does not match probe '
            f'geometry ({geometry.width_px}x{geometry.height_px} px); resample patterns first.'
        )

    detector_pixel_geometry = assembled_data.get_pixel_geometry()
    implied_pixel_width_m = (
        probe_wavelength_m * abs(detector_distance_m) / (width_px * detector_pixel_geometry.width_m)
    )
    implied_pixel_height_m = (
        probe_wavelength_m
        * abs(detector_distance_m)
        / (height_px * detector_pixel_geometry.height_m)
    )

    if not numpy.isclose(implied_pixel_width_m, geometry.pixel_width_m, rtol=rtol) or not (
        numpy.isclose(implied_pixel_height_m, geometry.pixel_height_m, rtol=rtol)
    ):
        raise ValueError(
            'Fresnel-transform output pixel size '
            f'({implied_pixel_width_m:.3e} x {implied_pixel_height_m:.3e} m) does not match '
            f'probe geometry ({geometry.pixel_width_m:.3e} x {geometry.pixel_height_m:.3e} m) '
            f'within rtol={rtol}.'
        )

    propagator_parameters = PropagatorParameters(
        wavelength_m=probe_wavelength_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=detector_pixel_geometry.width_m,
        pixel_height_m=detector_pixel_geometry.height_m,
        propagation_distance_m=-detector_distance_m,
    )
    propagator = FresnelTransformPropagator(propagator_parameters)
    array = propagator.propagate(numpy.sqrt(detector_intensity).astype(complex))

    return Probe(
        array=array,
        pixel_geometry=geometry.get_pixel_geometry(),
    )


@dataclass(frozen=True)
class FresnelZonePlate:
    """Physical parameters of a Fresnel zone plate optic."""

    zone_plate_diameter_m: float
    outermost_zone_width_m: float
    central_beamstop_diameter_m: float

    def get_focal_length_m(self, central_wavelength_m: float) -> float:
        """Return the zone plate focal length at *central_wavelength_m* (thin-lens formula)."""
        return self.zone_plate_diameter_m * self.outermost_zone_width_m / central_wavelength_m


def generate_fresnel_zone_plate_probe(
    geometry: ProbeGeometry,
    zone_plate: FresnelZonePlate,
    *,
    probe_wavelength_m: float,
    defocus_distance_m: float,
) -> Probe:
    """Simulate the probe formed by a Fresnel zone plate propagated to a given defocus distance."""
    focal_length_m = zone_plate.get_focal_length_m(probe_wavelength_m)
    propagation_distance_m = focal_length_m + defocus_distance_m

    fzp_plane_pixel_size_numerator = probe_wavelength_m * propagation_distance_m
    fzp_pixel_geometry = PixelGeometry(
        width_m=fzp_plane_pixel_size_numerator / geometry.width_m,
        height_m=fzp_plane_pixel_size_numerator / geometry.height_m,
    )

    # coordinate on FZP plane
    lx_fzp = -fzp_pixel_geometry.width_m * (
        numpy.arange(geometry.width_px) - geometry.width_px // 2
    )
    ly_fzp = -fzp_pixel_geometry.height_m * (
        numpy.arange(geometry.height_px) - geometry.height_px // 2
    )

    YY_FZP, XX_FZP = numpy.meshgrid(ly_fzp, lx_fzp, indexing='ij')  # noqa: N806
    RR_FZP = numpy.hypot(XX_FZP, YY_FZP)  # noqa: N806

    # transmission function of FZP
    T = numpy.exp(  # noqa: N806
        -2j * numpy.pi / probe_wavelength_m * (XX_FZP**2 + YY_FZP**2) / 2 / focal_length_m
    )
    C = RR_FZP <= zone_plate.zone_plate_diameter_m / 2  # noqa: N806
    H = RR_FZP >= zone_plate.central_beamstop_diameter_m / 2  # noqa: N806
    fzp_transmission_function = T * C * H

    propagator_parameters = PropagatorParameters(
        wavelength_m=probe_wavelength_m,
        width_px=fzp_transmission_function.shape[-1],
        height_px=fzp_transmission_function.shape[-2],
        pixel_width_m=fzp_pixel_geometry.width_m,
        pixel_height_m=fzp_pixel_geometry.height_m,
        propagation_distance_m=propagation_distance_m,
    )
    propagator = FresnelTransformPropagator(propagator_parameters)

    return Probe(
        array=propagator.propagate(fzp_transmission_function),
        pixel_geometry=geometry.get_pixel_geometry(),
    )


def generate_zernike_probe(
    geometry: ProbeGeometry, polynomial: Iterable[ZernikeMode], *, radius_m: float
) -> Probe:
    """Generate a probe as a superposition of Zernike polynomial modes within a circle of *radius_m*."""
    coords = geometry.get_transverse_coordinates()
    distance = coords.position_r_m / radius_m
    angle_rad = coords.angle_rad
    array = numpy.zeros_like(distance, dtype=complex)

    for mode in polynomial:
        array += mode(distance, angle_rad)

    return Probe(
        array=array,
        pixel_geometry=geometry.get_pixel_geometry(),
    )


def generate_hermite_probe(
    geometry: ProbeGeometry,
    polynomial: Iterable[HermiteMode],
    *,
    width_m: float,
    height_m: float,
) -> Probe:
    """Generate a probe as a superposition of 2D Hermite polynomial modes with characteristic widths *width_m* (x) and *height_m* (y)."""
    coords = geometry.get_transverse_coordinates()
    x = coords.position_x_m / width_m
    y = coords.position_y_m / height_m
    array = numpy.zeros_like(x, dtype=complex)

    for mode in polynomial:
        array += mode(x, y)

    return Probe(
        array=array,
        pixel_geometry=geometry.get_pixel_geometry(),
    )


def _random_phase_shift_axis(rng: numpy.random.Generator, size: int) -> ComplexArrayType:
    a = rng.uniform() - 0.5
    b = (size - 1 - 2 * numpy.arange(size)) / size
    return numpy.exp(1j * numpy.pi * a * b)


def generate_incoherent_probe_modes(
    rng: numpy.random.Generator,
    probe: Probe,
    imode_weights: Sequence[float],
    *,
    orthogonalize: bool = True,
) -> Probe:
    """Expand a probe to multiple incoherent modes with random phase shifts and relative weights."""
    num_imodes = len(imode_weights)
    array_in = probe.get_array()

    if numpy.isnan(array_in).any():
        logger.warning('Probe without incoherent modes contains NaN values!')
        return probe

    array_out_shape = num_imodes, *array_in.shape[-2:]
    array_out = numpy.zeros(array_out_shape, dtype=array_in.dtype)

    for imode in range(num_imodes):
        if imode < array_in.shape[-3]:
            # preserve existing incoherent modes
            values = array_in[imode, :, :]
        else:
            # apply random phase shift to 1st incoherent mode
            phase_shift_y = _random_phase_shift_axis(rng, array_in.shape[-2])
            phase_shift_x = _random_phase_shift_axis(rng, array_in.shape[-1])
            values = array_in[0, :, :] * numpy.outer(phase_shift_y, phase_shift_x)

        array_out[imode, :, :] = values

    if orthogonalize and num_imodes > 1:
        imodes_as_rows = array_out.reshape(num_imodes, -1)
        imodes_as_ortho_rows = scipy.linalg.orth(imodes_as_rows.T).T
        array_out = imodes_as_ortho_rows.reshape(array_out_shape)

    if numpy.isnan(array_out).any():
        logger.warning('Probe with incoherent modes contains NaN values!')
        return probe

    normalized_imode_weights = numpy.asarray(imode_weights) / numpy.sum(imode_weights)
    imode_intensity = numpy.sum(intensity(array_in)) * normalized_imode_weights

    for imode, intensity_out in enumerate(imode_intensity):
        intensity_in = numpy.sum(intensity(array_out[imode, :, :]))

        if intensity_in <= 0:
            logger.warning('Cannot rescale imode with zero intensity!')
        else:
            array_out[imode, :, :] *= numpy.sqrt(intensity_out / intensity_in)

    return Probe(
        array=array_out,
        pixel_geometry=probe.get_pixel_geometry(),
    )


def generate_coherent_probe_modes(
    rng: numpy.random.Generator,
    probe: Probe,
    *,
    num_cmodes: int,
    num_diffraction_patterns: int,
    small_value: float = 1.0e-6,
    normalize_cmodes: bool = True,
) -> ProbeSequence:
    """Build an OPR ProbeSequence with *num_cmodes* coherent modes and random per-scan weights."""
    opr_weights: RealArrayType | None = None

    if num_cmodes > 1:
        opr_weights = small_value * rng.normal(size=(num_diffraction_patterns, num_cmodes))
        opr_weights[:, 0] = 1.0

    array_in = probe.get_array()

    # Initialize every OPR mode (and its incoherent modes) with normalized Gaussian
    # random noise, then overwrite the main OPR mode with the input probe.
    array_out_shape = (num_cmodes, *array_in.shape)
    array_out = (rng.normal(size=array_out_shape) + 1j * rng.normal(size=array_out_shape)).astype(
        array_in.dtype
    )

    if normalize_cmodes:
        rms = numpy.sqrt(numpy.mean(intensity(array_out), axis=(-2, -1), keepdims=True))
        array_out /= rms

    array_out[0, :, :, :] = array_in[:, :, :]

    return ProbeSequence(
        array=array_out,
        opr_weights=opr_weights,
        pixel_geometry=probe.get_pixel_geometry(),
    )
