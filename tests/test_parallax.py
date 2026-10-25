"""Unit tests for estimate_defocus_from_parallax: defocus from shifted bright-field images.

Every scan is simulated by the far-field forward model, with a probe whose far field is a
uniform disk, propagated from its focus by the defocus under test. The sign of the
estimate is therefore pinned against that model rather than derived.

Behaviors verified:
  - a defocus beyond the depth of focus is recovered with its sign, pairing patterns
    with positions by index rather than by order
  - the depth of focus is the disk's ``lambda / NA**2 = 1 / (lambda * q_max**2)``, and
    the Fresnel number is the defocus in units of it
  - a probe at focus and a featureless sample are not resolved, and the depth-of-focus
    threshold is what rejects the probe at focus
  - too few bright-field bins give None; moments without binned patterns, or frames
    that do not match the probe, are rejected
"""

from functools import cache

import numpy
import pytest
from scipy.ndimage import gaussian_filter

from ptychodus.api.accumulate import DiffractionMoments, compute_diffraction_moments
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry
from ptychodus.api.parallax import ParallaxEstimate, estimate_defocus_from_parallax
from ptychodus.api.probe import Probe, ProbeEnsemble, ProbeGeometry
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.preprocess.diffraction import BinningStep
from ptychodus.api.simulate.probe import propagate_probe

_PIXEL_SIZE_M = 50e-9
_PROBE_PX = 64
_OBJECT_PX = 160
_DISK_RADIUS_PX = 8
_PIXEL_GEOMETRY = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
_PROBE_GEOMETRY = ProbeGeometry(
    width_px=_PROBE_PX,
    height_px=_PROBE_PX,
    pixel_width_m=_PIXEL_SIZE_M,
    pixel_height_m=_PIXEL_SIZE_M,
)
_OBJECT_GEOMETRY = ObjectGeometry(
    width_px=_OBJECT_PX,
    height_px=_OBJECT_PX,
    pixel_width_m=_PIXEL_SIZE_M,
    pixel_height_m=_PIXEL_SIZE_M,
    center_x_m=0.0,
    center_y_m=0.0,
)
_METADATA = ProductMetadata(
    name='parallax',
    comments='',
    detector_distance_m=2.0,
    photon_energy_eV=10_000.0,
    probe_photon_count=1.0,
    exposure_time_s=1.0,
    mass_attenuation_m2_per_kg=0.0,
    tomography_angle_deg=0.0,
)
_WAVELENGTH_M = _METADATA.photon_wavelength_m
# lambda / NA**2 for the disk: about 1.3 mm, so 2 mm is past it and 0 well inside.
_DEPTH_OF_FOCUS_M = 1.0 / (_WAVELENGTH_M * (_DISK_RADIUS_PX / (_PROBE_PX * _PIXEL_SIZE_M)) ** 2)


def _probe(defocus_m: float) -> Probe:
    q = numpy.arange(_PROBE_PX) - _PROBE_PX // 2
    disk = (numpy.hypot(q[:, numpy.newaxis], q) <= _DISK_RADIUS_PX).astype(complex)
    focus = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(disk)))
    probe = Probe(array=1000.0 * focus / numpy.abs(focus).max(), pixel_geometry=_PIXEL_GEOMETRY)
    return propagate_probe(
        probe, photon_wavelength_m=_WAVELENGTH_M, propagation_distance_m=defocus_m
    )


def _transmission(*, featureless: bool) -> numpy.ndarray:
    if featureless:
        return numpy.ones((_OBJECT_PX, _OBJECT_PX), dtype=complex)

    texture = gaussian_filter(
        numpy.random.default_rng(3).standard_normal((_OBJECT_PX, _OBJECT_PX)), 4.0
    )
    return numpy.exp((-0.05 + 0.5j) * texture / texture.std())


def _positions() -> list[ProbePosition]:
    rng = numpy.random.default_rng(1)
    positions: list[ProbePosition] = list()

    for iy in range(-36, 37, 4):
        for ix in range(-36, 37, 4):
            positions.append(
                ProbePosition(
                    index=len(positions),
                    x_m=(ix + rng.uniform(-0.5, 0.5)) * _PIXEL_SIZE_M,
                    y_m=(iy + rng.uniform(-0.5, 0.5)) * _PIXEL_SIZE_M,
                )
            )

    return positions


@cache
def _moments(
    defocus_m: float, *, featureless: bool = False, bin_size_px: int = 2
) -> DiffractionMoments:
    probe = _probe(defocus_m)
    product = Product(
        metadata=_METADATA,
        probe_positions=ProbePositionSequence(_positions()),
        probes=ProbeEnsemble(
            array=probe.get_array(), opr_weights=None, pixel_geometry=_PIXEL_GEOMETRY
        ),
        object_=Object(
            array=_transmission(featureless=featureless)[numpy.newaxis],
            pixel_geometry=_PIXEL_GEOMETRY,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )
    data = generate_diffraction_data(product, numpy.random.default_rng(0))
    moments = compute_diffraction_moments(
        numpy.array(data.get_indexes()),
        numpy.array(data.get_patterns()),
        bad_pixels=numpy.array(data.get_bad_pixels()),
        binning=BinningStep(bin_size_x=bin_size_px, bin_size_y=bin_size_px),
    )
    assert moments is not None
    return moments


def _estimate(
    moments: DiffractionMoments, positions: list[ProbePosition] | None = None
) -> ParallaxEstimate | None:
    return estimate_defocus_from_parallax(
        moments,
        ProbePositionSequence(_positions() if positions is None else positions),
        _OBJECT_GEOMETRY,
        _PROBE_GEOMETRY,
        photon_wavelength_m=_WAVELENGTH_M,
    )


@pytest.mark.parametrize('defocus_m', [2.0e-3, -2.0e-3])
def test_a_defocus_is_recovered_with_its_sign(defocus_m: float) -> None:
    # Reversed, so pairing by order rather than by index would scramble the images.
    estimate = _estimate(_moments(defocus_m), _positions()[::-1])

    assert estimate is not None
    # Measured within 4% at +-2 mm on this scan.
    assert estimate.defocus_distance_m == pytest.approx(defocus_m, rel=0.06)
    assert estimate.is_resolved()


def test_the_depth_of_focus_is_that_of_the_disk() -> None:
    estimate = _estimate(_moments(2.0e-3))

    assert estimate is not None
    # Second moments of the noisy pattern after its noise floor: measured within 4%.
    assert estimate.depth_of_focus_m == pytest.approx(_DEPTH_OF_FOCUS_M, rel=0.08)
    assert estimate.fresnel_number == pytest.approx(
        abs(estimate.defocus_distance_m) / estimate.depth_of_focus_m
    )


def test_a_probe_at_focus_is_not_resolved() -> None:
    estimate = _estimate(_moments(0.0))

    # Its images differ by phase contrast, not by a shift, so the fit finds a slope
    # with little scatter -- but no larger than the depth of focus.
    assert estimate is not None
    assert estimate.fresnel_number < 1.0
    assert estimate.significance > 3.0
    assert not estimate.is_resolved()
    assert estimate.is_resolved(minimum_defocus_dof=0.5 * estimate.fresnel_number)


def test_a_featureless_sample_is_not_resolved() -> None:
    estimate = _estimate(_moments(2.0e-3, featureless=True))

    # Its images are independent photon noise, so the slope is too: measured at about
    # two standard errors on this scan, against three needed.
    assert estimate is not None
    assert estimate.significance < 3.0
    assert not estimate.is_resolved()


def test_too_few_bright_field_bins_give_none() -> None:
    # Bins of 8 pixels leave a disk of radius 8 with no bin away from its edge.
    assert _estimate(_moments(2.0e-3, bin_size_px=8)) is None


def test_moments_without_binned_patterns_are_rejected() -> None:
    moments = _moments(2.0e-3)
    unbinned = DiffractionMoments(
        num_patterns=moments.num_patterns,
        minimum_frame=moments.minimum_frame,
        mean_frame=moments.mean_frame,
        maximum_frame=moments.maximum_frame,
        bad_pixels=moments.bad_pixels,
        indexes=moments.indexes,
        total_counts=moments.total_counts,
        center_of_mass_x_px=moments.center_of_mass_x_px,
        center_of_mass_y_px=moments.center_of_mass_y_px,
    )

    with pytest.raises(ValueError, match='binning step'):
        _estimate(unbinned)


def test_frames_that_do_not_match_the_probe_are_rejected() -> None:
    probe_geometry = ProbeGeometry(
        width_px=32, height_px=32, pixel_width_m=_PIXEL_SIZE_M, pixel_height_m=_PIXEL_SIZE_M
    )

    with pytest.raises(ValueError, match='do not match'):
        estimate_defocus_from_parallax(
            _moments(2.0e-3),
            ProbePositionSequence(_positions()),
            _OBJECT_GEOMETRY,
            probe_geometry,
            photon_wavelength_m=_WAVELENGTH_M,
        )
