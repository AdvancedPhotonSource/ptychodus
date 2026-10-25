"""Unit tests for estimate_probe_extent: real-space probe size from far-field second moments.

Each case builds the true probe from a far-field amplitude -- a disk, a rectangle or a
Gaussian -- back-transformed to its waist and carried a defocus downstream by
propagate_probe. Its far-field intensity, the mean pattern a sample-free scan records, is
what the estimator sees; the probe's own intensity moments are the truth it must recover.

Behaviors verified:
  - the beam widths of a Gaussian follow the second-moment law exactly, at any defocus
  - defocused disk and rectangle apertures recover their geometric shadow sizes
  - the sign of the defocus does not change the extent
  - a background pedestal and a masked hot pixel do not change it
  - zero defocus warns that only the transform-limited waist is left
  - the depth of focus is the disk's lambda / NA**2, and the Fresnel number is both the
    defocus in depths of focus and R**2 / (lambda |z|) of the shadow radius
"""

from typing import Any
import logging
import math

import numpy
import pytest

from ptychodus.api.moments import compute_intensity_moments
from ptychodus.api.probe import Probe, ProbeGeometry, estimate_probe_extent
from ptychodus.api.simulate.probe import propagate_probe

_NUM_PX = 256
_PIXEL_M = 10e-9
_WAVELENGTH_M = 1.0e-10
_DQ = 1.0 / (_NUM_PX * _PIXEL_M)
_GEOMETRY = ProbeGeometry(
    width_px=_NUM_PX, height_px=_NUM_PX, pixel_width_m=_PIXEL_M, pixel_height_m=_PIXEL_M
)


def _frequency_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    q = (numpy.arange(_NUM_PX) - _NUM_PX // 2) * _DQ
    return q.reshape(1, -1), q.reshape(-1, 1)


def _defocused_probe(far_field_amplitude: numpy.ndarray, defocus_m: float) -> Probe:
    waist = numpy.fft.fftshift(
        numpy.fft.ifft2(numpy.fft.ifftshift(far_field_amplitude.astype(complex)))
    )
    probe = Probe(array=waist, pixel_geometry=_GEOMETRY.get_pixel_geometry())
    return propagate_probe(
        probe, photon_wavelength_m=_WAVELENGTH_M, propagation_distance_m=defocus_m
    )


def _mean_pattern(probe: Probe) -> numpy.ndarray:
    field = probe.get_array()[0]
    return numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(numpy.fft.ifftshift(field)))) ** 2


def _true_sigmas_m(probe: Probe) -> tuple[float, float]:
    moments = compute_intensity_moments(numpy.abs(probe.get_array()[0]) ** 2)
    return (
        math.sqrt(float(moments.covariance_xx_px2)) * _PIXEL_M,
        math.sqrt(float(moments.covariance_yy_px2)) * _PIXEL_M,
    )


def test_a_gaussian_beam_follows_the_second_moment_law() -> None:
    qx, qy = _frequency_grid()
    sigma_q = 6.0 * _DQ
    amplitude = numpy.exp(-(qx**2 + qy**2) / (4.0 * sigma_q**2))

    for defocus_m in (0.0, 2.0e-4, 5.0e-4):
        probe = _defocused_probe(amplitude, defocus_m)
        true_x_m, true_y_m = _true_sigmas_m(probe)
        extent = estimate_probe_extent(
            _mean_pattern(probe),
            _GEOMETRY,
            photon_wavelength_m=_WAVELENGTH_M,
            defocus_distance_m=defocus_m,
        )

        assert extent.beam_sigma_x_m == pytest.approx(true_x_m, rel=0.01)
        assert extent.beam_sigma_y_m == pytest.approx(true_y_m, rel=0.01)
        assert extent.gaussian_fwhm_m == pytest.approx(
            2.0 * math.sqrt(2.0 * math.log(2.0)) * true_x_m, rel=0.01
        )


@pytest.mark.parametrize('defocus_m', [5.0e-4, -5.0e-4])
def test_a_defocused_disk_recovers_its_diameter(defocus_m: float) -> None:
    qx, qy = _frequency_grid()
    edge_q = 20.0 * _DQ
    amplitude = (numpy.hypot(qx, qy) <= edge_q).astype(numpy.float64)
    probe = _defocused_probe(amplitude, defocus_m)

    extent = estimate_probe_extent(
        _mean_pattern(probe),
        _GEOMETRY,
        photon_wavelength_m=_WAVELENGTH_M,
        defocus_distance_m=defocus_m,
    )

    geometric_diameter_m = 2.0 * _WAVELENGTH_M * abs(defocus_m) * edge_q
    assert extent.disk_diameter_m == pytest.approx(geometric_diameter_m, rel=0.02)
    # The beam's own second moment also carries the diffraction tails around the shadow.
    true_x_m, _ = _true_sigmas_m(probe)
    assert extent.disk_diameter_m / 4.0 < true_x_m


def test_a_defocused_rectangle_recovers_its_sides() -> None:
    qx, qy = _frequency_grid()
    half_width_q = 24.0 * _DQ
    half_height_q = 12.0 * _DQ
    amplitude = ((numpy.abs(qx) <= half_width_q) & (numpy.abs(qy) <= half_height_q)).astype(float)
    defocus_m = 4.0e-4
    probe = _defocused_probe(amplitude, defocus_m)

    extent = estimate_probe_extent(
        _mean_pattern(probe),
        _GEOMETRY,
        photon_wavelength_m=_WAVELENGTH_M,
        defocus_distance_m=defocus_m,
    )

    # The aperture spans 2 * half + 1 pixels of the frequency grid.
    assert extent.rectangle_width_m == pytest.approx(
        _WAVELENGTH_M * defocus_m * (2.0 * half_width_q + _DQ), rel=0.01
    )
    assert extent.rectangle_height_m == pytest.approx(
        _WAVELENGTH_M * defocus_m * (2.0 * half_height_q + _DQ), rel=0.01
    )


def test_a_pedestal_and_a_masked_hot_pixel_do_not_change_the_extent() -> None:
    qx, qy = _frequency_grid()
    amplitude = (numpy.hypot(qx, qy) <= 20.0 * _DQ).astype(numpy.float64)
    pattern = _mean_pattern(_defocused_probe(amplitude, 5.0e-4))
    kwargs: dict[str, Any] = dict(photon_wavelength_m=_WAVELENGTH_M, defocus_distance_m=5.0e-4)
    reference = estimate_probe_extent(pattern, _GEOMETRY, **kwargs)

    corrupted = pattern + 0.01 * pattern.max()
    corrupted[3, 250] = 100.0 * pattern.max()
    bad_pixels = numpy.zeros(pattern.shape, dtype=bool)
    bad_pixels[3, 250] = True

    extent = estimate_probe_extent(corrupted, _GEOMETRY, bad_pixels=bad_pixels, **kwargs)

    assert extent.disk_diameter_m == pytest.approx(reference.disk_diameter_m, rel=0.01)


def test_zero_defocus_warns_that_only_the_waist_is_left(
    caplog: pytest.LogCaptureFixture,
) -> None:
    qx, qy = _frequency_grid()
    amplitude = (numpy.hypot(qx, qy) <= 20.0 * _DQ).astype(numpy.float64)

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.probe'):
        extent = estimate_probe_extent(
            _mean_pattern(_defocused_probe(amplitude, 0.0)),
            _GEOMETRY,
            photon_wavelength_m=_WAVELENGTH_M,
            defocus_distance_m=0.0,
        )

    assert any('transform-limited' in record.message for record in caplog.records)
    sigma_q = 10.0 * _DQ  # a uniform disk of radius 20 dq has sigma_q = 10 dq
    waist_m = 1.0 / (4.0 * math.pi * sigma_q)
    assert extent.beam_sigma_m == pytest.approx(waist_m, rel=0.03)
    assert extent.disk_diameter_m == pytest.approx(4.0 * waist_m, rel=0.03)


@pytest.mark.parametrize('defocus_m', [5.0e-4, -2.0e-4])
def test_the_depth_of_focus_and_fresnel_number_are_the_disks(defocus_m: float) -> None:
    qx, qy = _frequency_grid()
    edge_q = 20.0 * _DQ
    amplitude = (numpy.hypot(qx, qy) <= edge_q).astype(numpy.float64)

    extent = estimate_probe_extent(
        _mean_pattern(_defocused_probe(amplitude, defocus_m)),
        _GEOMETRY,
        photon_wavelength_m=_WAVELENGTH_M,
        defocus_distance_m=defocus_m,
    )

    # lambda / NA**2 with NA = lambda * edge_q.
    assert extent.depth_of_focus_m == pytest.approx(1.0 / (_WAVELENGTH_M * edge_q**2), rel=0.02)
    assert extent.fresnel_number == pytest.approx(abs(defocus_m) / extent.depth_of_focus_m)
    shadow_radius_m = extent.disk_diameter_m / 2.0
    assert extent.fresnel_number == pytest.approx(
        shadow_radius_m**2 / (_WAVELENGTH_M * abs(defocus_m)), rel=1e-12
    )


def test_a_mismatched_pattern_is_rejected() -> None:
    with pytest.raises(ValueError, match='does not match'):
        estimate_probe_extent(
            numpy.ones((32, 32)),
            _GEOMETRY,
            photon_wavelength_m=_WAVELENGTH_M,
            defocus_distance_m=1.0e-4,
        )
