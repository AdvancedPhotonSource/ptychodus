"""Probe generation functions: geometric apertures, zone plates, KB mirrors, Zernike modes, Hermite modes, and OPR ensembles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum, auto
import logging
import math

import numpy
import numpy.polynomial.legendre
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..constants import TWO_PI, TWO_PI_J
from ..typing import ComplexArrayType, RealArrayType
from ..geometry import HermiteMode, ImageExtent, LegendreMode, PixelGeometry, ZernikeMode
from ..probe import Probe, ProbeGeometry, ProbeSequence, ProbeTransverseCoordinates
from ..propagate import (
    AngularSpectrumPropagator,
    FresnelTransformPropagator,
    Propagator,
    PropagatorParameters,
    compute_far_field_pixel_geometry,
    intensity,
)
from ..assemble import AssembledDiffractionData


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


def propagate_probe(
    probe: Probe,
    *,
    photon_wavelength_m: float,
    propagation_distance_m: float,
) -> Probe:
    """Propagate every incoherent mode of a probe by *propagation_distance_m*.

    Uses the angular-spectrum method, which is exact for all Fresnel numbers. The
    propagator acts on the last two axes, so the mode axis rides through untouched and
    every mode sees the same transfer function.
    """
    pixel_geometry = probe.get_pixel_geometry()
    propagator_parameters = PropagatorParameters(
        wavelength_m=photon_wavelength_m,
        width_px=probe.width_px,
        height_px=probe.height_px,
        pixel_width_m=pixel_geometry.width_m,
        pixel_height_m=pixel_geometry.height_m,
        propagation_distance_m=propagation_distance_m,
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
        numpy.fabs(coords.x_m) < 0.5 * width_m,
        numpy.fabs(coords.y_m) < 0.5 * height_m,
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
    photon_wavelength_m: float,
    detector_distance_m: float,
    far_field: bool = True,
    rtol: float = 1.0e-3,
) -> Probe:
    """Back-propagate the square root of the mean diffraction pattern to estimate the probe.

    Far field back-propagates with the single-FFT Fresnel transform, whose output pitch
    is ``lambda * |z| / (N * dx_det)``; that pitch must match what *geometry* declares
    for the returned Probe to be self-consistent, so it is checked against *rtol*.

    Near field back-propagates with the angular spectrum, which preserves pitch, so the
    probe lands on the grid *geometry* already describes and there is no output pitch to
    check. The sample grid is the detector grid demagnified, which pins the magnification
    at ``dx_det / dx_probe`` and the equivalent parallel-beam distance at ``z_d / M``.

    Raises ValueError if the diffraction-pattern shape is inconsistent with *geometry*,
    if the detector distance is zero, or -- far field only -- if the implied
    sample-plane pixel size disagrees with *geometry*.
    """
    if detector_distance_m == 0.0:
        raise ValueError(
            'Detector distance must be nonzero to back-propagate the average pattern; '
            'the Fresnel-transform output pixel size lambda*z/(N*dx_det) vanishes at z=0.'
        )

    detector_intensity = numpy.mean(assembled_data.get_patterns(), axis=0)
    height_px, width_px = detector_intensity.shape[-2:]

    if (width_px, height_px) != (geometry.width_px, geometry.height_px):
        raise ValueError(
            f'Diffraction pattern shape ({width_px}x{height_px} px) does not match probe '
            f'geometry ({geometry.width_px}x{geometry.height_px} px); resample patterns first.'
        )

    detector_pixel_geometry = assembled_data.get_pixel_geometry()
    probe_pixel_geometry = geometry.get_pixel_geometry()

    if far_field:
        implied_geometry = ProbeGeometry.from_far_field(
            detector_pixel_geometry,
            ImageExtent(width_px=width_px, height_px=height_px),
            wavelength_m=photon_wavelength_m,
            distance_m=detector_distance_m,
        )

        if not numpy.isclose(
            implied_geometry.pixel_width_m, geometry.pixel_width_m, rtol=rtol
        ) or not numpy.isclose(implied_geometry.pixel_height_m, geometry.pixel_height_m, rtol=rtol):
            raise ValueError(
                'Fresnel-transform output pixel size '
                f'({implied_geometry.pixel_width_m:.3e} x '
                f'{implied_geometry.pixel_height_m:.3e} m) does not match '
                f'probe geometry ({geometry.pixel_width_m:.3e} x '
                f'{geometry.pixel_height_m:.3e} m) within rtol={rtol}.'
            )

        # Backward propagation, so PropagatorParameters' pitch describes the *output*
        # (upstream) plane -- the sample plane, not the detector.
        propagator_parameters = PropagatorParameters(
            wavelength_m=photon_wavelength_m,
            width_px=width_px,
            height_px=height_px,
            pixel_width_m=probe_pixel_geometry.width_m,
            pixel_height_m=probe_pixel_geometry.height_m,
            propagation_distance_m=-detector_distance_m,
        )
        propagator: Propagator = FresnelTransformPropagator(propagator_parameters)
    else:
        # The sample grid is the detector grid demagnified, so the pitch ratio is the
        # magnification and no separate output-pitch check is possible or needed:
        # angular spectrum returns the field on the grid it was given.
        magnification = detector_pixel_geometry.width_m / probe_pixel_geometry.width_m
        propagator_parameters = PropagatorParameters(
            wavelength_m=photon_wavelength_m,
            width_px=width_px,
            height_px=height_px,
            pixel_width_m=probe_pixel_geometry.width_m,
            pixel_height_m=probe_pixel_geometry.height_m,
            propagation_distance_m=-detector_distance_m / magnification,
        )
        propagator = AngularSpectrumPropagator(propagator_parameters)

    array = propagator.propagate(numpy.sqrt(detector_intensity).astype(complex))

    return Probe(array=array, pixel_geometry=probe_pixel_geometry)


class FresnelZonePlate(BaseModel):
    """Physical parameters of a Fresnel zone plate optic."""

    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)

    zone_plate_diameter_m: float = Field(gt=0.0)
    """Diameter of the outermost zone."""
    outermost_zone_width_m: float = Field(gt=0.0)
    """Width of the outermost zone, which sets the diffraction-limited resolution."""
    central_beamstop_diameter_m: float = Field(ge=0.0)
    """Diameter of the central stop; zero means the zone plate has none."""

    @model_validator(mode='after')
    def _validate_beamstop_fits(self) -> FresnelZonePlate:
        # The pupil is the annulus between the stop and the outer zone, so a stop that
        # reaches the rim leaves nothing to illuminate with.
        if self.central_beamstop_diameter_m >= self.zone_plate_diameter_m:
            raise ValueError(
                f'Central beamstop diameter ({self.central_beamstop_diameter_m}) must be '
                f'smaller than the zone plate diameter ({self.zone_plate_diameter_m}); '
                'otherwise the zone plate passes no light at all.'
            )

        return self

    def get_focal_length_m(self, central_wavelength_m: float) -> float:
        """Return the zone plate focal length at *central_wavelength_m* (thin-lens formula)."""
        return self.zone_plate_diameter_m * self.outermost_zone_width_m / central_wavelength_m

    def get_numerical_aperture(self, central_wavelength_m: float) -> float:
        """Return the half-angle the outer zone subtends at the focus.

        ``D / (2 f)``, which reduces to ``lambda / (2 dr_N)`` -- the convergence aperture
        implied by :attr:`outermost_zone_width_m` alone, independent of the diameter.
        Counterpart of :meth:`KirkpatrickBaezMirror.get_numerical_aperture`, and distinct
        from the detector's collection aperture.

        Raises:
            ZeroDivisionError: at zero wavelength, where the focal length is undefined.
        """
        return 0.5 * self.zone_plate_diameter_m / self.get_focal_length_m(central_wavelength_m)


def generate_fresnel_zone_plate_probe(
    geometry: ProbeGeometry,
    zone_plate: FresnelZonePlate,
    *,
    photon_wavelength_m: float,
    defocus_distance_m: float,
) -> Probe:
    """Simulate the probe formed by a Fresnel zone plate propagated to a given defocus distance."""
    focal_length_m = zone_plate.get_focal_length_m(photon_wavelength_m)
    propagation_distance_m = focal_length_m + defocus_distance_m

    if propagation_distance_m == 0.0:
        raise ValueError(
            'Zone plate focal length plus defocus distance must be nonzero; the '
            'Fresnel-transform pixel size lambda*z/(N*dx) vanishes at z=0.'
        )

    probe_pixel_geometry = geometry.get_pixel_geometry()
    fzp_pixel_geometry = compute_far_field_pixel_geometry(
        probe_pixel_geometry,
        ImageExtent(width_px=geometry.width_px, height_px=geometry.height_px),
        wavelength_m=photon_wavelength_m,
        propagation_distance_m=propagation_distance_m,
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
        -TWO_PI_J / photon_wavelength_m * (XX_FZP**2 + YY_FZP**2) / 2 / focal_length_m
    )
    C = RR_FZP <= zone_plate.zone_plate_diameter_m / 2  # noqa: N806
    H = RR_FZP >= zone_plate.central_beamstop_diameter_m / 2  # noqa: N806
    fzp_transmission_function = T * C * H

    # PropagatorParameters' pitch describes the upstream plane: the zone plate when the
    # probe plane lies downstream of it, and the probe plane itself when the distance is
    # negative and the propagation runs backward.
    upstream_pixel_geometry = (
        fzp_pixel_geometry if propagation_distance_m > 0.0 else probe_pixel_geometry
    )
    propagator_parameters = PropagatorParameters(
        wavelength_m=photon_wavelength_m,
        width_px=fzp_transmission_function.shape[-1],
        height_px=fzp_transmission_function.shape[-2],
        pixel_width_m=upstream_pixel_geometry.width_m,
        pixel_height_m=upstream_pixel_geometry.height_m,
        propagation_distance_m=propagation_distance_m,
    )
    propagator = FresnelTransformPropagator(propagator_parameters)

    return Probe(
        array=propagator.propagate(fzp_transmission_function),
        pixel_geometry=probe_pixel_geometry,
    )


# The figure-error and focus-sampling warnings below pull in opposite directions: a
# FWHM of the sinc focus a uniformly illuminated slit of numerical aperture NA forms,
# as a multiple of lambda / (2 NA).
_SINC_FWHM_FACTOR = 0.886


class KirkpatrickBaezMirror(BaseModel):
    """Physical parameters of one grazing-incidence focusing mirror of a KB pair."""

    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)

    acceptance_length_m: float = Field(gt=0.0)
    """Illuminated length along the mirror surface."""
    grazing_angle_rad: float = Field(gt=0.0, lt=0.5 * numpy.pi)
    """Angle between the incident beam and the mirror surface.

    Bounded below normal incidence, which keeps ``sin(theta)`` positive so the projected
    and numerical apertures stay positive; past that it is not a grazing-incidence
    mirror. Real values are milliradians.
    """
    focus_distance_m: float = Field(gt=0.0)
    """Distance from the mirror center to the nominal focus."""
    source_distance_m: float = Field(default=0.0, ge=0.0)
    """Distance from the source to the mirror center. Zero means a collimated input."""

    @classmethod
    def from_numerical_aperture(
        cls,
        numerical_aperture: float,
        *,
        focus_distance_m: float,
        grazing_angle_rad: float,
        source_distance_m: float = 0.0,
    ) -> KirkpatrickBaezMirror:
        """Build a mirror from the numerical aperture rather than the acceptance length.

        Inverts :meth:`get_numerical_aperture`, so the acceptance length that comes back
        is the one that subtends *numerical_aperture* at *focus_distance_m* when projected
        through *grazing_angle_rad*.
        """
        projected_aperture_m = 2.0 * numerical_aperture * focus_distance_m
        return cls(
            acceptance_length_m=float(projected_aperture_m / numpy.sin(grazing_angle_rad)),
            grazing_angle_rad=grazing_angle_rad,
            focus_distance_m=focus_distance_m,
            source_distance_m=source_distance_m,
        )

    def get_projected_aperture_m(self) -> float:
        """Return the acceptance length projected normal to the beam: ``L sin(theta)``."""
        return float(self.acceptance_length_m * numpy.sin(self.grazing_angle_rad))

    def get_numerical_aperture(self) -> float:
        """Return the half-angle the projected aperture subtends at the focus."""
        return 0.5 * self.get_projected_aperture_m() / self.focus_distance_m

    def get_focal_length_m(self) -> float:
        """Return the focal length ``p q / (p + q)``.

        A zero source distance denotes collimated input, where the focal length is the
        focus distance itself.
        """
        if self.source_distance_m == 0.0:
            return self.focus_distance_m

        return (
            self.source_distance_m
            * self.focus_distance_m
            / (self.source_distance_m + self.focus_distance_m)
        )


class KirkpatrickBaezMirrorPair(BaseModel):
    """A Kirkpatrick-Baez pair: one mirror focusing in x, one focusing in y."""

    model_config = ConfigDict(frozen=True, extra='forbid', allow_inf_nan=False)

    horizontal: KirkpatrickBaezMirror
    """Mirror that focuses in the x-direction."""
    vertical: KirkpatrickBaezMirror
    """Mirror that focuses in the y-direction."""

    def get_reference_distance_m(self) -> float:
        """Return the pupil reference distance: the mean of the two mirror-focus distances.

        The two mirrors sit at different distances from a shared focal plane, so no single
        plane is *the* pupil. The mean is a convenient stand-in: the beam each mirror
        defines has half-width ``NA z`` at any plane z, so the choice changes the pupil
        sampling but not the field at the focus.
        """
        return 0.5 * (self.horizontal.focus_distance_m + self.vertical.focus_distance_m)


def generate_mirror_figure_error(
    rng: numpy.random.Generator,
    *,
    rms_slope_error_rad: float,
    acceptance_length_m: float,
    num_modes: int = 12,
    lowest_order: int = 2,
    psd_exponent: float = -2.0,
) -> list[LegendreMode]:
    """Draw a random mirror figure error with a power-law spectrum and a given rms slope.

    Coefficient magnitudes are drawn proportional to ``order ** (psd_exponent / 2)``, so
    *psd_exponent* sets how fast the height spectrum falls with spatial frequency; more
    negative values concentrate the error in low orders. *num_modes* sets how many terms
    are returned and *lowest_order* where they start -- the default of 2 omits piston and
    tilt, matching the metrology convention of quoting figure error after removing them.
    The whole coefficient vector is then scaled so its rms slope over the aperture equals
    *rms_slope_error_rad*.

    Raises ValueError for a non-positive mode count, a negative lowest order, a negative
    rms slope error, or a non-positive acceptance length.
    """
    if num_modes < 1:
        raise ValueError(f'Mode count must be at least one (got {num_modes})!')

    if lowest_order < 0:
        raise ValueError(f'Lowest order must be non-negative (got {lowest_order})!')

    if rms_slope_error_rad < 0.0:
        raise ValueError(f'RMS slope error must be non-negative (got {rms_slope_error_rad})!')

    if acceptance_length_m <= 0.0:
        raise ValueError(f'Acceptance length must be positive (got {acceptance_length_m})!')

    orders = numpy.arange(lowest_order, lowest_order + num_modes)

    # Order zero is piston, whose power-law weight would diverge and whose slope is
    # identically zero; clamping keeps it finite and lets the renormalization below
    # ignore it on its own.
    spectrum = numpy.power(numpy.maximum(orders, 1), 0.5 * psd_exponent)
    coefficients = rng.normal(size=num_modes) * spectrum

    # Legendre polynomials are orthogonal on [-1, 1] but their derivatives are not, so
    # the rms slope of a sum is not the quadrature sum of per-order slopes. Differentiate
    # the assembled series instead and rescale once.
    legendre_series = numpy.zeros(lowest_order + num_modes)
    legendre_series[lowest_order:] = coefficients
    slope_series = numpy.polynomial.legendre.legder(legendre_series)

    # Gauss-Legendre is exact for the squared derivative, whose degree is at most
    # 2 (lowest_order + num_modes) - 2, so this many nodes integrates it without error.
    nodes, weights = numpy.polynomial.legendre.leggauss(lowest_order + num_modes)
    slope_per_u = numpy.polynomial.legendre.legval(nodes, slope_series)

    # du spans the aperture over [-1, 1], so arc length along the mirror is s = L u / 2.
    slope_per_s = 2.0 * slope_per_u / acceptance_length_m
    mean_square_slope = 0.5 * numpy.sum(weights * numpy.square(slope_per_s))

    if mean_square_slope <= 0.0:
        logger.warning('Cannot scale a figure error whose slope vanishes everywhere!')
        return [LegendreMode(coefficient_m=0.0, order=int(order)) for order in orders]

    scale = rms_slope_error_rad / numpy.sqrt(mean_square_slope)

    return [
        LegendreMode(coefficient_m=float(coefficient * scale), order=int(order))
        for coefficient, order in zip(coefficients, orders)
    ]


def _compute_figure_error_phase(
    u: RealArrayType,
    modes: Iterable[LegendreMode],
    *,
    photon_wavelength_m: float,
    grazing_angle_rad: float,
) -> RealArrayType:
    """Return the reflected phase imposed by a surface figure error on a grazing mirror.

    A height *h* normal to the surface shortens the reflected path by ``2 h sin(theta)``,
    which advances the wavefront -- hence the negative sign, matching the sign the
    converging pupil phase uses for an advanced wavefront.
    """
    height_m = numpy.zeros_like(u)

    for mode in modes:
        height_m = height_m + mode(u)

    return -2.0 * TWO_PI / photon_wavelength_m * height_m * numpy.sin(grazing_angle_rad)


def generate_kb_mirror_probe(
    geometry: ProbeGeometry,
    mirrors: KirkpatrickBaezMirrorPair,
    *,
    photon_wavelength_m: float,
    defocus_distance_m: float = 0.0,
    astigmatism_m: float = 0.0,
    incident_beam_fwhm_x_m: float = 0.0,
    incident_beam_fwhm_y_m: float = 0.0,
    figure_error_x: Iterable[LegendreMode] = (),
    figure_error_y: Iterable[LegendreMode] = (),
    min_pupil_px_for_figure_error: float = 16.0,
    min_px_per_focus_fwhm: float = 2.0,
) -> Probe:
    """Simulate the probe a Kirkpatrick-Baez mirror pair forms at a given defocus distance.

    The separable rectangular pupil is built on the plane conjugate to *geometry*, so the
    aperture is a real optic dimension and the focus lands critically sampled on the probe
    grid. Each axis gets the numerical aperture its mirror defines.

    *astigmatism_m* separates the two axes' foci along z, placing the x focus half of it
    upstream of the reference plane and the y focus half of it downstream. A per-axis
    propagation distance is algebraically the same as a residual quadratic pupil phase, so
    one propagation carries both axes exactly.

    *incident_beam_fwhm_x_m* and *incident_beam_fwhm_y_m* give the intensity FWHM of the
    beam illuminating the pupil, in projected pupil coordinates; zero means uniform
    illumination, whose hard edges produce far stronger sinc side lobes than a real mirror.
    *figure_error_x* and *figure_error_y* carry each mirror's surface height error over its
    own normalized aperture coordinate.

    *min_pupil_px_for_figure_error* and *min_px_per_focus_fwhm* set when the grid is
    called too coarse to warn about. Raising the first demands a better-resolved pupil
    before figure error is taken seriously; raising the second demands more probe pixels
    across the focus. They pull in opposite directions -- a finer probe pitch shrinks the
    pupil footprint while a coarser one undersamples the focus -- and both scale with the
    array size, so a grid too small satisfies neither. Only logging depends on them.

    Raises ValueError when the propagation distance vanishes, when either focus plane
    coincides with the pupil, or when a projected aperture does not fit inside the pupil
    window the conjugate grid provides.
    """
    # Materialize once: an Iterable may be a one-shot generator, and the modes are
    # both tested for emptiness and evaluated below.
    figure_error_x_modes = tuple(figure_error_x)
    figure_error_y_modes = tuple(figure_error_y)

    reference_distance_m = mirrors.get_reference_distance_m()
    propagation_distance_m = reference_distance_m + defocus_distance_m

    if propagation_distance_m == 0.0:
        raise ValueError(
            'Mirror reference distance plus defocus distance must be nonzero; the '
            'Fresnel-transform pixel size lambda*z/(N*dx) vanishes at z=0.'
        )

    focus_distance_x_m = reference_distance_m - 0.5 * astigmatism_m
    focus_distance_y_m = reference_distance_m + 0.5 * astigmatism_m

    if focus_distance_x_m == 0.0 or focus_distance_y_m == 0.0:
        raise ValueError(
            'Astigmatism places one axis focus at the pupil plane, where the converging '
            'phase pi x^2 / (lambda z) diverges; keep it below twice the reference '
            f'distance ({2 * reference_distance_m:.3e} m).'
        )

    probe_pixel_geometry = geometry.get_pixel_geometry()
    image_extent = ImageExtent(width_px=geometry.width_px, height_px=geometry.height_px)
    pupil_pixel_geometry = compute_far_field_pixel_geometry(
        probe_pixel_geometry,
        image_extent,
        wavelength_m=photon_wavelength_m,
        propagation_distance_m=propagation_distance_m,
    )

    # Centering matches PropagatorParameters.get_spatial_coordinates so that an odd
    # figure-error order shifts the focus the way the propagator's own grid implies.
    lx_pupil = pupil_pixel_geometry.width_m * (
        numpy.arange(geometry.width_px) - geometry.width_px // 2
    )
    ly_pupil = pupil_pixel_geometry.height_m * (
        numpy.arange(geometry.height_px) - geometry.height_px // 2
    )
    YY_PUPIL, XX_PUPIL = numpy.meshgrid(ly_pupil, lx_pupil, indexing='ij')  # noqa: N806

    numerical_aperture_x = mirrors.horizontal.get_numerical_aperture()
    numerical_aperture_y = mirrors.vertical.get_numerical_aperture()
    half_aperture_x_m = numerical_aperture_x * reference_distance_m
    half_aperture_y_m = numerical_aperture_y * reference_distance_m

    pupil_width_m = geometry.width_px * pupil_pixel_geometry.width_m
    pupil_height_m = geometry.height_px * pupil_pixel_geometry.height_m

    if 2.0 * half_aperture_x_m > pupil_width_m or 2.0 * half_aperture_y_m > pupil_height_m:
        raise ValueError(
            f'Projected aperture ({2 * half_aperture_x_m:.3e} x '
            f'{2 * half_aperture_y_m:.3e} m) does not fit the pupil window '
            f'({pupil_width_m:.3e} x {pupil_height_m:.3e} m); the numerical aperture '
            'would be clipped. The window is lambda*z/dx_probe, so use a finer probe '
            'pixel size; the array size does not enter.'
        )

    _warn_about_kb_sampling(
        numerical_aperture_x=numerical_aperture_x,
        numerical_aperture_y=numerical_aperture_y,
        half_aperture_x_m=half_aperture_x_m,
        half_aperture_y_m=half_aperture_y_m,
        pupil_pixel_geometry=pupil_pixel_geometry,
        probe_pixel_geometry=probe_pixel_geometry,
        photon_wavelength_m=photon_wavelength_m,
        has_figure_error=bool(figure_error_x_modes) or bool(figure_error_y_modes),
        min_pupil_px_for_figure_error=min_pupil_px_for_figure_error,
        min_px_per_focus_fwhm=min_px_per_focus_fwhm,
    )

    is_inside = numpy.logical_and(
        numpy.fabs(XX_PUPIL) <= half_aperture_x_m,
        numpy.fabs(YY_PUPIL) <= half_aperture_y_m,
    )
    amplitude = numpy.where(is_inside, 1.0, 0.0)

    # The FWHM describes the intensity, so the amplitude falls off half as fast.
    if incident_beam_fwhm_x_m > 0.0:
        amplitude = amplitude * numpy.exp(
            -2.0 * numpy.log(2.0) * numpy.square(XX_PUPIL / incident_beam_fwhm_x_m)
        )

    if incident_beam_fwhm_y_m > 0.0:
        amplitude = amplitude * numpy.exp(
            -2.0 * numpy.log(2.0) * numpy.square(YY_PUPIL / incident_beam_fwhm_y_m)
        )

    phase = (
        -numpy.pi
        / photon_wavelength_m
        * (
            numpy.square(XX_PUPIL) / focus_distance_x_m
            + numpy.square(YY_PUPIL) / focus_distance_y_m
        )
    )
    phase = phase + _compute_figure_error_phase(
        XX_PUPIL / half_aperture_x_m,
        figure_error_x_modes,
        photon_wavelength_m=photon_wavelength_m,
        grazing_angle_rad=mirrors.horizontal.grazing_angle_rad,
    )
    phase = phase + _compute_figure_error_phase(
        YY_PUPIL / half_aperture_y_m,
        figure_error_y_modes,
        photon_wavelength_m=photon_wavelength_m,
        grazing_angle_rad=mirrors.vertical.grazing_angle_rad,
    )

    kb_transmission_function = amplitude * numpy.exp(1j * phase)

    # PropagatorParameters' pitch describes the upstream plane: the pupil when the probe
    # plane lies downstream of it, and the probe plane itself when the distance is
    # negative and the propagation runs backward.
    upstream_pixel_geometry = (
        pupil_pixel_geometry if propagation_distance_m > 0.0 else probe_pixel_geometry
    )
    propagator_parameters = PropagatorParameters(
        wavelength_m=photon_wavelength_m,
        width_px=kb_transmission_function.shape[-1],
        height_px=kb_transmission_function.shape[-2],
        pixel_width_m=upstream_pixel_geometry.width_m,
        pixel_height_m=upstream_pixel_geometry.height_m,
        propagation_distance_m=propagation_distance_m,
    )
    propagator = FresnelTransformPropagator(propagator_parameters)

    return Probe(
        array=propagator.propagate(kb_transmission_function),
        pixel_geometry=probe_pixel_geometry,
    )


def _warn_about_kb_sampling(
    *,
    numerical_aperture_x: float,
    numerical_aperture_y: float,
    half_aperture_x_m: float,
    half_aperture_y_m: float,
    pupil_pixel_geometry: PixelGeometry,
    probe_pixel_geometry: PixelGeometry,
    photon_wavelength_m: float,
    has_figure_error: bool,
    min_pupil_px_for_figure_error: float,
    min_px_per_focus_fwhm: float,
) -> None:
    """Warn when the grid resolves either the pupil or the focus too coarsely."""
    if has_figure_error:
        pupil_px_x = 2.0 * half_aperture_x_m / pupil_pixel_geometry.width_m
        pupil_px_y = 2.0 * half_aperture_y_m / pupil_pixel_geometry.height_m
        fewest_pupil_px = min(pupil_px_x, pupil_px_y)

        if fewest_pupil_px < min_pupil_px_for_figure_error:
            logger.warning(
                'Projected aperture spans only %.1f pupil pixels; figure error is '
                'barely representable below %.1f. Use a finer probe pixel size.',
                fewest_pupil_px,
                min_pupil_px_for_figure_error,
            )

    focus_px_x = (
        _SINC_FWHM_FACTOR
        * photon_wavelength_m
        / (2.0 * numerical_aperture_x * probe_pixel_geometry.width_m)
    )
    focus_px_y = (
        _SINC_FWHM_FACTOR
        * photon_wavelength_m
        / (2.0 * numerical_aperture_y * probe_pixel_geometry.height_m)
    )
    fewest_focus_px = min(focus_px_x, focus_px_y)

    if fewest_focus_px < min_px_per_focus_fwhm:
        logger.warning(
            'Focus spans only %.1f probe pixels FWHM; below %.1f it is undersampled. '
            'Use a coarser probe pixel size.',
            fewest_focus_px,
            min_px_per_focus_fwhm,
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
    x = coords.x_m / width_m
    y = coords.y_m / height_m
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


def _gram_schmidt(rows: ComplexArrayType, *, dependence_floor: float) -> ComplexArrayType:
    """Orthonormalize *rows* in their original order by modified Gram-Schmidt.

    Unlike an SVD basis for the same span, this keeps row k paired with input row k, so a
    caller that attached meaning to the ordering -- a mode index, an eigenvalue -- still
    has it afterward. There is deliberately no column pivoting: it would sharpen rank
    detection at the cost of reordering the rows, which is the one property this exists
    to provide.

    A row is dropped to zero when the part of it orthogonal to its predecessors falls
    below *dependence_floor* times its own length. The comparison has to be relative: a
    dependent row leaves a residual that is roundoff rather than zero, so an absolute
    test against zero accepts it, scales it to unit length, and returns a vector of pure
    noise that is not orthogonal to anything.

    Each row is projected twice. One pass loses orthogonality in proportion to the
    conditioning of the input; the second restores it to roundoff, which holds even for
    inputs conditioned at the limit of the format.

    Returns double precision whatever it is given.
    """
    # Work at full precision with every row scaled to unit peak. numpy.linalg.norm sums
    # squares without guarding the exponent, so a row far from unit scale sends both
    # norms below to zero or to infinity and the row is then misread as dependent. The
    # scaling cancels out: the test below is a ratio of two norms of the same row, and
    # the rows that come back are normalized.
    work = rows.astype(numpy.complex128, copy=True)
    peak = numpy.abs(work).max(axis=-1, keepdims=True)
    work = numpy.divide(work, peak, out=numpy.zeros_like(work), where=peak > 0.0)

    orthonormal_rows = numpy.zeros_like(work)

    for k in range(work.shape[0]):
        residual = work[k].copy()
        original_norm = numpy.linalg.norm(residual)

        for _ in range(2):
            for j in range(k):
                residual -= numpy.vdot(orthonormal_rows[j], residual) * orthonormal_rows[j]

        norm = numpy.linalg.norm(residual)

        if norm > dependence_floor * original_norm:
            orthonormal_rows[k] = residual / norm
        else:
            logger.warning('Dropping an incoherent mode that is dependent on its predecessors!')

    return orthonormal_rows


class ProbeModeDecayType(IntEnum):
    """How power falls off across a sequence of incoherent probe modes."""

    NONE = auto()
    """All power in the first mode; every later mode gets none."""
    POLYNOMIAL = auto()
    """Mode n carries ``(n + 1) ** log2(decay_ratio)`` of the first mode's power."""
    EXPONENTIAL = auto()
    """Mode n carries ``decay_ratio ** n`` of the first mode's power."""

    def get_weights(self, num_modes: int, decay_ratio: float) -> Sequence[float]:
        """Return unnormalized relative power for *num_modes* modes.

        *decay_ratio* is the power of the second mode relative to the first, and must be
        positive for every type but :attr:`NONE`. A ratio of one makes the weights
        uniform; a ratio above one makes later modes outrank earlier ones.
        """
        match self:
            case ProbeModeDecayType.EXPONENTIAL:
                b = 1.0 / decay_ratio
                return [b**-n for n in range(num_modes)]
            case ProbeModeDecayType.POLYNOMIAL:
                b = math.log(decay_ratio) / math.log(2.0)
                return [(n + 1) ** b for n in range(num_modes)]
            case _:
                return [1.0] + [0.0] * (num_modes - 1)


def _validate_decay_ratio(decay_type: ProbeModeDecayType, decay_ratio: float) -> None:
    """Reject a decay ratio that would make the weights negative, infinite or undefined."""
    if decay_type is ProbeModeDecayType.NONE:
        return

    if not math.isfinite(decay_ratio) or decay_ratio <= 0.0:
        raise ValueError(f'Decay ratio must be positive and finite (got {decay_ratio})!')


def _probe_geometry(probe: Probe) -> ProbeGeometry:
    """Return the grid geometry a single probe occupies."""
    pixel_geometry = probe.get_pixel_geometry()
    return ProbeGeometry(
        width_px=probe.width_px,
        height_px=probe.height_px,
        pixel_width_m=pixel_geometry.width_m,
        pixel_height_m=pixel_geometry.height_m,
    )


class IncoherentModeStrategy(ABC):
    """Policy for the modes a probe gains when it is expanded.

    A strategy answers two questions: what shape the new modes take, and how power would
    divide across a mode sequence of a given length. Everything else -- preserving the
    modes a probe already carries, reconciling the count, dividing the power budget and
    orthogonalizing the result -- is the same for every strategy and belongs to
    :func:`generate_incoherent_probe_modes`.
    """

    @abstractmethod
    def get_imode_weights(self, num_imodes: int) -> RealArrayType:
        """Return unnormalized relative power for a sequence of *num_imodes* modes."""

    @abstractmethod
    def build_fill_imodes(self, probe: Probe, first_imode: int, num_fill: int) -> ComplexArrayType:
        """Return *num_fill* new mode shapes to occupy indices *first_imode* onward.

        The shapes are unweighted and need not be orthogonal, either to each other or to
        the modes *probe* already carries.
        """


class RandomPhaseRampStrategy(IncoherentModeStrategy):
    """Fill with copies of the dominant mode carrying random separable phase ramps.

    The new modes start with the dominant mode's amplitude structure and differ only by a
    random linear phase ramp in x and in y, which orthogonalization then spreads across
    the set. Nothing about the construction predicts how power should divide, so the
    weights come from a decay profile.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        *,
        decay_type: ProbeModeDecayType = ProbeModeDecayType.EXPONENTIAL,
        decay_ratio: float = 0.5,
    ) -> None:
        _validate_decay_ratio(decay_type, decay_ratio)
        self._rng = rng
        self._decay_type = decay_type
        self._decay_ratio = decay_ratio

    def get_imode_weights(self, num_imodes: int) -> RealArrayType:
        return numpy.asarray(
            self._decay_type.get_weights(num_imodes, self._decay_ratio), dtype=float
        )

    def build_fill_imodes(self, probe: Probe, first_imode: int, num_fill: int) -> ComplexArrayType:
        array_in = probe.get_array()
        dominant_mode = array_in[0, :, :].astype(complex)

        return numpy.stack(
            [
                dominant_mode
                * numpy.outer(
                    _random_phase_shift_axis(self._rng, array_in.shape[-2]),
                    _random_phase_shift_axis(self._rng, array_in.shape[-1]),
                )
                for _ in range(num_fill)
            ]
        )


def _gaussian_schell_axis(beam_size_m: float, coherence_length_m: float) -> tuple[float, float]:
    """Return the coherent-mode envelope parameter c and the mode weight ratio r.

    Standard Gaussian-Schell algebra for one axis: with ``a = 1/(4 sigma^2)`` from the
    intensity width and ``b = 1/(2 xi^2)`` from the coherence length,
    ``c = sqrt(a^2 + 2ab)`` sets the envelope ``exp(-c x^2)`` of every eigenmode and
    ``r = b/(a + b + c)`` makes the eigenvalue spectrum the geometric series ``r^n``.
    Note c exceeds a whenever the beam is partially coherent, so each individual mode is
    narrower than the total intensity profile.
    """
    a = 1.0 / (4.0 * numpy.square(beam_size_m))
    b = 1.0 / (2.0 * numpy.square(coherence_length_m))
    c = numpy.sqrt(numpy.square(a) + 2.0 * a * b)
    return float(c), float(b / (a + b + c))


class GaussianSchellStrategy(IncoherentModeStrategy):
    """Fill with the Hermite-Gauss eigenmodes of a Gaussian-Schell source.

    The mode set is separable in x and y, so the two-dimensional weights are the outer
    product of the two per-axis geometric series and the strongest are kept. Unlike the
    other strategies the weights are predicted from the source rather than taken from a
    decay profile, which is the point of the model.

    Mode ``(m, n)`` is the dominant mode of the probe multiplied by
    ``H_m(x sqrt(2 c_x)) H_n(y sqrt(2 c_y))`` -- the Hermite factor of the eigenfunction
    without its Gaussian envelope, since the probe already carries an envelope of its
    own. That makes the construction exact when the probe's envelope is the
    ``exp(-c x^2)`` the model predicts, and a reasonable localized basis otherwise.

    The predicted spectrum describes this strategy's own eigenmodes. A probe that already
    carries modes keeps them, and they are not those, so the weights on the preserved
    indices become approximate whenever the input has more than one mode.
    """

    def __init__(
        self,
        *,
        beam_size_x_m: float,
        coherence_length_x_m: float,
        beam_size_y_m: float,
        coherence_length_y_m: float,
    ) -> None:
        for name, value in (
            ('beam_size_x_m', beam_size_x_m),
            ('coherence_length_x_m', coherence_length_x_m),
            ('beam_size_y_m', beam_size_y_m),
            ('coherence_length_y_m', coherence_length_y_m),
        ):
            if value <= 0.0:
                raise ValueError(f'{name} must be positive (got {value})!')

        self._envelope_x, self._ratio_x = _gaussian_schell_axis(beam_size_x_m, coherence_length_x_m)
        self._envelope_y, self._ratio_y = _gaussian_schell_axis(beam_size_y_m, coherence_length_y_m)

    def _get_strongest(self, num_imodes: int) -> tuple[list[tuple[int, int]], RealArrayType]:
        # Both per-axis series decrease in their own order, so the strongest num_imodes
        # products cannot involve an order at or beyond num_imodes on either axis.
        orders_x, orders_y = numpy.meshgrid(
            numpy.arange(num_imodes), numpy.arange(num_imodes), indexing='ij'
        )
        weights = numpy.power(self._ratio_x, orders_x) * numpy.power(self._ratio_y, orders_y)
        strongest = numpy.argsort(weights, axis=None, kind='stable')[::-1][:num_imodes]
        orders = [(int(orders_x.flat[i]), int(orders_y.flat[i])) for i in strongest]
        return orders, weights.flat[strongest]

    def get_imode_orders(self, num_imodes: int) -> list[tuple[int, int]]:
        """Return the Hermite order pair ``(m, n)`` backing each mode, strongest first."""
        return self._get_strongest(num_imodes)[0]

    def get_imode_weights(self, num_imodes: int) -> RealArrayType:
        return self._get_strongest(num_imodes)[1]

    def build_fill_imodes(self, probe: Probe, first_imode: int, num_fill: int) -> ComplexArrayType:
        array_in = probe.get_array()
        coords = _probe_geometry(probe).get_transverse_coordinates()
        x = coords.x_m * numpy.sqrt(2.0 * self._envelope_x)
        y = coords.y_m * numpy.sqrt(2.0 * self._envelope_y)
        dominant_mode = array_in[0, :, :].astype(complex)
        orders = self.get_imode_orders(first_imode + num_fill)[first_imode:]

        return numpy.stack(
            [
                dominant_mode * HermiteMode(1.0 + 0j, order_x, order_y)(x, y)
                for order_x, order_y in orders
            ]
        )


@dataclass(frozen=True)
class _ProbeMoments:
    """Intensity-weighted centroid and variance of a probe, in meters."""

    center_x_m: float
    center_y_m: float
    variance_x_m2: float
    variance_y_m2: float


def _compute_probe_moments(
    probe_intensity: RealArrayType, coordinates: ProbeTransverseCoordinates
) -> _ProbeMoments:
    """Return the plain intensity-weighted first and second moments of a probe.

    No filtering, thresholding or background subtraction: the caller wants the length
    scale of the distribution it passed in, not an estimate of where a cleaned-up beam
    would sit.

    Raises ValueError when the total intensity is non-positive, or when either variance
    vanishes, since neither a centroid nor a width is defined in those cases.
    """
    total = probe_intensity.sum()

    if total <= 0.0:
        raise ValueError('Cannot take moments of a probe with non-positive total intensity!')

    center_x_m = (coordinates.x_m * probe_intensity).sum() / total
    center_y_m = (coordinates.y_m * probe_intensity).sum() / total
    variance_x_m2 = (numpy.square(coordinates.x_m - center_x_m) * probe_intensity).sum() / total
    variance_y_m2 = (numpy.square(coordinates.y_m - center_y_m) * probe_intensity).sum() / total

    if variance_x_m2 <= 0.0 or variance_y_m2 <= 0.0:
        raise ValueError(
            'Probe intensity has no transverse extent on at least one axis, so it sets no '
            'length scale; a probe confined to a single row or column cannot seed '
            'polynomial modes.'
        )

    return _ProbeMoments(
        center_x_m=float(center_x_m),
        center_y_m=float(center_y_m),
        variance_x_m2=float(variance_x_m2),
        variance_y_m2=float(variance_y_m2),
    )


def _graded_polynomial_orders(num_orders: int) -> list[tuple[int, int]]:
    """Return *num_orders* polynomial order pairs, graded by increasing total degree.

    Every order of total degree d precedes every order of degree d + 1, so truncating
    the list keeps the lowest-degree terms: ``(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)``.
    """
    orders: list[tuple[int, int]] = []
    degree = 0

    while len(orders) < num_orders:
        for order_x in range(degree, -1, -1):
            orders.append((order_x, degree - order_x))

            if len(orders) == num_orders:
                break

        degree += 1

    return orders


class ProbeMomentPolynomialStrategy(IncoherentModeStrategy):
    """Fill with damped polynomials scaled by the probe's own intensity moments.

    Alone among the strategies here, this one needs no physical parameters: it measures
    the intensity-weighted centroid and variance of the probe it is handed and uses that
    width to set its own length scale. The same call therefore suits any illumination --
    a focused optic, a pinhole, a back-propagated mean pattern -- without being told
    anything about how the probe was formed. The moments come from the incoherent sum of
    the probe's modes, which is the footprint the illumination actually covers.

    A fill mode is ``u**m * v**n`` times the dominant mode, where u and v are the
    transverse coordinates centered on the probe and expressed in units of its rms width,
    damped by a Gaussian of that same width. The polynomial factor alone would push each
    successive mode further into the tails; the damping holds the set inside the region
    the probe occupies. Orders are graded by total degree and skip the constant term,
    which would only reproduce the dominant mode.

    Weights come from a decay profile and depend on a mode's index, not its order, so
    when a count truncates part-way through a degree the surviving modes are not ranked
    among themselves by anything physical.

    *damping_width* scales the Gaussian in units of the probe's rms width; the default of
    one makes the first-order modes come out the same width as the dominant mode, while
    larger values damp less and let the higher orders spread.
    """

    def __init__(
        self,
        *,
        decay_type: ProbeModeDecayType = ProbeModeDecayType.EXPONENTIAL,
        decay_ratio: float = 0.5,
        damping_width: float = 1.0,
    ) -> None:
        _validate_decay_ratio(decay_type, decay_ratio)

        if damping_width <= 0.0:
            raise ValueError(f'Damping width must be positive (got {damping_width})!')

        self._decay_type = decay_type
        self._decay_ratio = decay_ratio
        self._damping_width = damping_width

    def get_imode_weights(self, num_imodes: int) -> RealArrayType:
        return numpy.asarray(
            self._decay_type.get_weights(num_imodes, self._decay_ratio), dtype=float
        )

    def build_fill_imodes(self, probe: Probe, first_imode: int, num_fill: int) -> ComplexArrayType:
        array_in = probe.get_array()
        coords = _probe_geometry(probe).get_transverse_coordinates()
        moments = _compute_probe_moments(numpy.sum(intensity(array_in), axis=-3), coords)

        # Dimensionless offsets: a polynomial in meters would span tens of decades
        # between its lowest and highest order for no gain, since scaling a mode by a
        # constant leaves the orthogonalized result unchanged.
        u = (coords.x_m - moments.center_x_m) / math.sqrt(moments.variance_x_m2)
        v = (coords.y_m - moments.center_y_m) / math.sqrt(moments.variance_y_m2)
        damping = numpy.exp(
            -(numpy.square(u) + numpy.square(v)) / (2.0 * numpy.square(self._damping_width))
        )
        dominant_mode = array_in[0, :, :].astype(complex)
        orders = _graded_polynomial_orders(num_fill + 1)[1:]

        return numpy.stack(
            [
                numpy.power(u, order_x) * numpy.power(v, order_y) * dominant_mode * damping
                for order_x, order_y in orders
            ]
        )


DEFAULT_INCOHERENT_MODE_STRATEGY = ProbeMomentPolynomialStrategy()
"""The strategy :func:`generate_incoherent_probe_modes` uses when none is named."""


def _split_imode_budget(
    profile: RealArrayType, measured: RealArrayType, num_existing: int
) -> RealArrayType:
    """Divide the power budget between preserved modes and newly built ones.

    The new modes take the share the strategy's profile puts on their indices; the modes
    the probe arrived with divide the rest in the proportions they arrived with, so a
    converged mixed state keeps its character instead of being flattened onto a profile
    that was never meant to describe it.
    """
    normalized_profile = profile / numpy.sum(profile)
    fill_budget = float(numpy.sum(normalized_profile[num_existing:]))

    weights = numpy.empty_like(normalized_profile)
    weights[:num_existing] = measured * (1.0 - fill_budget)
    weights[num_existing:] = normalized_profile[num_existing:]
    return weights


def _measured_imode_powers(probe: Probe) -> RealArrayType:
    """Return each existing mode's share of the probe's intensity, summing to one."""
    powers = numpy.asarray(
        [
            probe.get_incoherent_mode_relative_power(imode)
            for imode in range(probe.num_incoherent_modes)
        ],
        dtype=float,
    )
    total = numpy.sum(powers)

    if total <= 0.0:
        return numpy.full(len(powers), 1.0 / len(powers))

    return powers / total


def generate_incoherent_probe_modes(
    probe: Probe,
    num_imodes: int,
    *,
    strategy: IncoherentModeStrategy = DEFAULT_INCOHERENT_MODE_STRATEGY,
    orthogonalize: bool = True,
    mode_dependence_floor: float = 1.0e-8,
) -> Probe:
    """Return *probe* carried onto a sequence of *num_imodes* mutually incoherent modes.

    Modes the probe already carries are kept, in order, and *strategy* supplies only the
    shortfall; asking for fewer modes than the probe has keeps the strongest. Either way
    the probe's total power is unchanged, because the number of modes used to represent a
    beam is a modeling choice while its illumination is not.

    Preserved modes keep the share of the power they arrived with, and the new modes
    divide the share the strategy's profile assigns to their indices.

    *mode_dependence_floor* is the fraction of its own length a mode's orthogonal part
    must keep to be retained; raising it discards marginally independent modes sooner.
    """
    if num_imodes < 1:
        raise ValueError(f'Mode count must be at least one (got {num_imodes})!')

    array_in = probe.get_array()

    if numpy.isnan(array_in).any():
        logger.warning('Probe without incoherent modes contains NaN values!')
        return probe

    num_existing = array_in.shape[-3]
    measured = _measured_imode_powers(probe)

    if num_imodes < num_existing:
        strongest = numpy.argsort(measured, kind='stable')[::-1][:num_imodes]
        array_out = array_in[strongest, :, :].astype(complex)
        weights: RealArrayType = measured[strongest] / numpy.sum(measured[strongest])
    else:
        array_out = array_in.astype(complex)
        num_fill = num_imodes - num_existing

        if num_fill > 0:
            fill = strategy.build_fill_imodes(probe, num_existing, num_fill)
            array_out = numpy.concatenate([array_out, fill.astype(complex)])

        weights = _split_imode_budget(
            strategy.get_imode_weights(num_imodes), measured, num_existing
        )

    if orthogonalize and num_imodes > 1:
        array_out_shape = array_out.shape
        array_out = _gram_schmidt(
            array_out.reshape(num_imodes, -1), dependence_floor=mode_dependence_floor
        ).reshape(array_out_shape)

    if array_out.shape[-3] != len(weights):
        raise ValueError(
            f'Built {array_out.shape[-3]} incoherent mode(s) against {len(weights)} '
            'weight(s); the two must agree.'
        )

    if numpy.isnan(array_out).any():
        logger.warning('Probe with incoherent modes contains NaN values!')
        return probe

    # A mode the orthogonalization dropped cannot carry power, so its share goes back to
    # the modes that survived rather than leaving the probe dimmer than it arrived.
    surviving = numpy.asarray(
        [numpy.sum(intensity(values)) > 0.0 for values in array_out], dtype=bool
    )

    if not surviving.any():
        logger.warning('Every incoherent mode was dropped; keeping the probe unchanged!')
        return probe

    weights = numpy.where(surviving, weights, 0.0)
    weights = weights / numpy.sum(weights)
    imode_intensity = numpy.sum(intensity(array_in)) * weights

    for imode, intensity_out in enumerate(imode_intensity):
        intensity_in = numpy.sum(intensity(array_out[imode, :, :]))

        if intensity_in > 0.0:
            array_out[imode, :, :] *= numpy.sqrt(intensity_out / intensity_in)

    return Probe(
        array=array_out.astype(array_in.dtype),
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
