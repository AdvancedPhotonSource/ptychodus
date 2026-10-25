"""Unit tests for the intensity-moment primitive and the shared noise-floor front end.

Behaviors verified:
  - compute_intensity_moments measures centroids from the array midpoint and the central
    covariance of known distributions (a point, a uniform disk, a tilted Gaussian)
  - it works over a leading batch axis, and a masked pixel carries no weight
  - an empty image has a zero total and NaN moments rather than raising
  - subtract_noise_floor removes a pedestal and keeps the full height of the signal above it
  - with keep_filtered=False the measured heights survive, only the mask is filtered
  - estimate_beam_center ignores a hot pixel once it is masked
  - reconcile_defocus_distance_m lets a declared focus win, and flips the sign
"""

import logging

import numpy
import pytest

from ptychodus.api.diffraction import BadPixels
from ptychodus.api.moments import compute_intensity_moments
from ptychodus.api.preprocess.diffraction import estimate_beam_center
from ptychodus.api.product import reconcile_defocus_distance_m
from ptychodus.api.preprocess.noise import (
    estimate_filtered_noise_floor,
    subtract_noise_floor,
)


def _centered_grid(height: int, width: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    y = numpy.arange(height, dtype=numpy.float64).reshape(-1, 1) - (height - 1) / 2
    x = numpy.arange(width, dtype=numpy.float64).reshape(1, -1) - (width - 1) / 2
    return y, x


def test_a_single_pixel_sits_at_its_offset_from_the_midpoint() -> None:
    image = numpy.zeros((9, 12))
    image[2, 10] = 3.0

    moments = compute_intensity_moments(image)

    assert float(moments.total) == 3.0
    assert float(moments.centroid_x_px) == pytest.approx(10 - 5.5)
    assert float(moments.centroid_y_px) == pytest.approx(2 - 4.0)
    assert float(moments.covariance_xx_px2) == pytest.approx(0.0, abs=1e-12)
    assert float(moments.covariance_yy_px2) == pytest.approx(0.0, abs=1e-12)


def test_a_uniform_disk_has_variance_a_quarter_of_its_squared_radius() -> None:
    radius_px = 40.0
    y, x = _centered_grid(256, 256)
    disk = (numpy.hypot(x, y) <= radius_px).astype(numpy.float64)

    moments = compute_intensity_moments(disk)

    assert float(moments.centroid_x_px) == pytest.approx(0.0, abs=1e-9)
    assert float(moments.centroid_y_px) == pytest.approx(0.0, abs=1e-9)
    assert float(moments.covariance_xx_px2) == pytest.approx(radius_px**2 / 4, rel=0.02)
    assert float(moments.covariance_yy_px2) == pytest.approx(radius_px**2 / 4, rel=0.02)
    assert float(moments.covariance_xy_px2) == pytest.approx(0.0, abs=1e-9)


def test_a_tilted_gaussian_recovers_its_covariance() -> None:
    covariance = numpy.array([[30.0, 12.0], [12.0, 18.0]])
    y, x = _centered_grid(128, 160)
    x = x - 7.0
    y = y + 3.0
    inverse = numpy.linalg.inv(covariance)
    exponent = inverse[0, 0] * x**2 + 2 * inverse[0, 1] * x * y + inverse[1, 1] * y**2
    image = numpy.exp(-0.5 * exponent)

    moments = compute_intensity_moments(image)

    assert float(moments.centroid_x_px) == pytest.approx(7.0, abs=1e-6)
    assert float(moments.centroid_y_px) == pytest.approx(-3.0, abs=1e-6)
    assert float(moments.covariance_xx_px2) == pytest.approx(30.0, rel=1e-6)
    assert float(moments.covariance_xy_px2) == pytest.approx(12.0, rel=1e-6)
    assert float(moments.covariance_yy_px2) == pytest.approx(18.0, rel=1e-6)


def test_moments_run_over_a_leading_batch_axis() -> None:
    rng = numpy.random.default_rng(0)
    stack = rng.uniform(0.0, 1.0, size=(5, 3, 16, 20))

    batched = compute_intensity_moments(stack)

    assert batched.total.shape == (5, 3)

    for index in numpy.ndindex(5, 3):
        single = compute_intensity_moments(stack[index])
        assert batched.centroid_x_px[index] == pytest.approx(float(single.centroid_x_px))
        assert batched.covariance_xy_px2[index] == pytest.approx(float(single.covariance_xy_px2))


def test_a_masked_pixel_carries_no_weight() -> None:
    y, x = _centered_grid(64, 64)
    image = numpy.exp(-(x**2 + y**2) / 50.0)
    hot = image.copy()
    hot[3, 60] = 1.0e6
    bad_pixels = numpy.zeros(image.shape, dtype=bool)
    bad_pixels[3, 60] = True

    clean = compute_intensity_moments(image, bad_pixels=bad_pixels)
    masked = compute_intensity_moments(hot, bad_pixels=bad_pixels)
    unmasked = compute_intensity_moments(hot)

    assert float(masked.centroid_x_px) == pytest.approx(float(clean.centroid_x_px))
    assert float(masked.covariance_yy_px2) == pytest.approx(float(clean.covariance_yy_px2))
    assert abs(float(unmasked.centroid_x_px)) > 10.0


def test_an_empty_image_has_nan_moments() -> None:
    moments = compute_intensity_moments(numpy.zeros((8, 8)))

    assert float(moments.total) == 0.0
    assert numpy.isnan(moments.centroid_x_px)
    assert numpy.isnan(moments.covariance_xx_px2)


def test_subtracting_the_floor_removes_a_pedestal_and_keeps_the_signal() -> None:
    y, x = _centered_grid(128, 128)
    signal = 1000.0 * (numpy.hypot(x, y) <= 20.0)
    rng = numpy.random.default_rng(1)
    pattern = rng.poisson(signal + 50.0).astype(numpy.float64)

    cleaned, statistics = subtract_noise_floor(pattern)

    assert statistics.median == pytest.approx(50.0, abs=2.0)
    outside = cleaned[numpy.hypot(x, y) > 25.0].sum()
    inside = cleaned[numpy.hypot(x, y) < 15.0].sum()
    assert outside < 1.0e-3 * inside
    # Full height above the background, not lowered by the significance margin.
    assert numpy.median(cleaned[numpy.hypot(x, y) < 15.0]) == pytest.approx(1000.0, rel=0.02)


def test_unfiltered_heights_keep_the_measured_values_under_the_mask() -> None:
    rng = numpy.random.default_rng(6)
    pattern = rng.normal(10.0, 1.0, size=(48, 48))
    pattern[12:36, 12:36] += rng.uniform(500.0, 1500.0, size=(24, 24))

    filtered, statistics = subtract_noise_floor(pattern)
    measured, _ = subtract_noise_floor(pattern, keep_filtered=False)

    interior = (slice(13, 35), slice(13, 35))
    numpy.testing.assert_allclose(measured[interior], pattern[interior] - statistics.median)
    assert not numpy.allclose(filtered[interior], measured[interior])
    assert numpy.count_nonzero(measured[:8, :]) == 0


def test_a_masked_pixel_stays_out_of_the_noise_statistics() -> None:
    rng = numpy.random.default_rng(2)
    pattern = rng.normal(10.0, 1.0, size=(64, 64))
    pattern[:8, :] = 1.0e4
    bad_pixels = numpy.zeros(pattern.shape, dtype=bool)
    bad_pixels[:8, :] = True

    _, masked = estimate_filtered_noise_floor(pattern, bad_pixels=bad_pixels)

    assert masked.median == pytest.approx(10.0, abs=0.5)


def test_subtracting_the_floor_rejects_a_negative_margin() -> None:
    with pytest.raises(ValueError):
        subtract_noise_floor(numpy.ones((8, 8)), mad_threshold=-1.0)


def test_beam_center_ignores_a_masked_hot_pixel() -> None:
    y, x = _centered_grid(96, 96)
    pattern = 500.0 * numpy.exp(-((x - 6.0) ** 2 + (y + 4.0) ** 2) / 30.0) + 5.0
    pattern[2:6, 85:89] = 1.0e7
    bad_pixels: BadPixels = numpy.zeros((96, 96), dtype=bool)
    bad_pixels[2:6, 85:89] = True

    center = estimate_beam_center(pattern, bad_pixels=bad_pixels)

    assert (center.x_px, center.y_px) == (round(47.5 + 6.0), round(47.5 - 4.0))


def test_a_declared_focus_wins_and_flips_sign(caplog: pytest.LogCaptureFixture) -> None:
    assert reconcile_defocus_distance_m(0.0, 8.0e-4) == 8.0e-4
    assert reconcile_defocus_distance_m(-8.0e-4, 0.0) == pytest.approx(8.0e-4)
    assert reconcile_defocus_distance_m(8.0e-4, 0.0) == pytest.approx(-8.0e-4)

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.product'):
        assert reconcile_defocus_distance_m(-8.0e-4, 8.0e-4) == pytest.approx(8.0e-4)
        assert not caplog.records
        assert reconcile_defocus_distance_m(-8.0e-4, 9.0e-4) == pytest.approx(8.0e-4)

    assert len(caplog.records) == 1
