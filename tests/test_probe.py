"""Unit tests for ptychodus.api.probe.estimate_probe_size."""

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import (
    Probe,
    ProbeSizeMetrics,
    compute_shannon_entropy,
    estimate_probe_entropy,
    estimate_probe_size,
)


PIXEL_M = 1e-9  # 1 nm per pixel — keeps lengths interpretable as "px == nm"
PIXEL_GEOMETRY = PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M)


def _gaussian_2d(
    height: int,
    width: int,
    sigma_major_px: float,
    sigma_minor_px: float,
    tilt_rad: float,
) -> numpy.ndarray:
    """Construct a tilted anisotropic 2D Gaussian intensity on a centered grid."""
    y_idx, x_idx = numpy.mgrid[:height, :width]
    dx = x_idx - (width - 1) / 2.0
    dy = y_idx - (height - 1) / 2.0
    cos_t = numpy.cos(tilt_rad)
    sin_t = numpy.sin(tilt_rad)
    dx_p = cos_t * dx + sin_t * dy  # along the requested major axis
    dy_p = -sin_t * dx + cos_t * dy  # along the requested minor axis
    return numpy.exp(-0.5 * ((dx_p / sigma_major_px) ** 2 + (dy_p / sigma_minor_px) ** 2))


class TestEstimateProbeSize:
    def test_isotropic_gaussian_recovers_2sigma_fwhm_and_encircled_diameter(self) -> None:
        """For a known Gaussian, all three widths should match analytic values."""
        sigma_px = 10.0
        intensity = _gaussian_2d(128, 128, sigma_px, sigma_px, tilt_rad=0.0)

        metrics = estimate_probe_size(intensity, PIXEL_GEOMETRY)

        expected_2sigma = 2.0 * sigma_px * PIXEL_M
        expected_fwhm = 2.0 * numpy.sqrt(2.0 * numpy.log(2.0)) * sigma_px * PIXEL_M
        # 2D Gaussian encircled-energy radius at fraction f: r = sigma * sqrt(-2 ln(1 - f))
        expected_encircled = 2.0 * sigma_px * numpy.sqrt(-2.0 * numpy.log(0.2)) * PIXEL_M

        numpy.testing.assert_allclose(metrics.rms_major_axis_length_m, expected_2sigma, rtol=0.02)
        numpy.testing.assert_allclose(metrics.rms_minor_axis_length_m, expected_2sigma, rtol=0.02)
        numpy.testing.assert_allclose(metrics.fwhm_major_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(metrics.fwhm_minor_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(
            metrics.encircled_energy_diameter_m, expected_encircled, rtol=0.05
        )

    def test_rms_and_fwhm_share_full_width_scale(self) -> None:
        """Now that RMS is reported as the full 2-sigma width, the FWHM / RMS ratio
        for a Gaussian should be sqrt(2 ln 2) ~ 1.1774 — *not* 2*sqrt(2 ln 2) ~ 2.355.
        """
        intensity = _gaussian_2d(128, 128, sigma_major_px=10.0, sigma_minor_px=10.0, tilt_rad=0.0)
        metrics = estimate_probe_size(intensity, PIXEL_GEOMETRY)

        ratio = metrics.fwhm_major_axis_length_m / metrics.rms_major_axis_length_m
        numpy.testing.assert_allclose(ratio, numpy.sqrt(2.0 * numpy.log(2.0)), rtol=0.03)

    def test_anisotropic_tilted_gaussian_round_trip(self) -> None:
        """Major/minor axis lengths, ordering, and tilt should all round-trip."""
        sigma_major_px = 12.0
        sigma_minor_px = 6.0
        tilt = numpy.pi / 6.0  # 30 deg, comfortably inside [-pi/2, pi/2)
        intensity = _gaussian_2d(128, 128, sigma_major_px, sigma_minor_px, tilt)

        metrics = estimate_probe_size(intensity, PIXEL_GEOMETRY)

        assert metrics.rms_major_axis_length_m > metrics.rms_minor_axis_length_m
        assert metrics.fwhm_major_axis_length_m > metrics.fwhm_minor_axis_length_m

        numpy.testing.assert_allclose(
            metrics.rms_major_axis_length_m, 2.0 * sigma_major_px * PIXEL_M, rtol=0.02
        )
        numpy.testing.assert_allclose(
            metrics.rms_minor_axis_length_m, 2.0 * sigma_minor_px * PIXEL_M, rtol=0.02
        )

        numpy.testing.assert_allclose(metrics.major_axis_tilt_rad, tilt, atol=0.05)
        # Minor axis is perpendicular to major: |sin(major - minor)| ~ 1.
        delta = metrics.major_axis_tilt_rad - metrics.minor_axis_tilt_rad
        numpy.testing.assert_allclose(abs(numpy.sin(delta)), 1.0, atol=0.02)

    def test_uniform_disk_recovers_radius_and_encircled_diameter(self) -> None:
        """For a uniform disk of radius R: sigma = R/2 (so 2-sigma = R),
        the 1D projection is a semicircle profile with FWHM = R*sqrt(3),
        and the 80%-encircled diameter is 2 R sqrt(0.8).
        """
        radius_px = 20.0
        size = 128
        y_idx, x_idx = numpy.mgrid[:size, :size]
        r_px = numpy.hypot(x_idx - (size - 1) / 2.0, y_idx - (size - 1) / 2.0)
        disk = (r_px <= radius_px).astype(numpy.float64)

        metrics = estimate_probe_size(disk, PIXEL_GEOMETRY)

        expected_2sigma = radius_px * PIXEL_M
        expected_fwhm = radius_px * numpy.sqrt(3.0) * PIXEL_M
        expected_encircled = 2.0 * radius_px * numpy.sqrt(0.8) * PIXEL_M

        numpy.testing.assert_allclose(metrics.rms_major_axis_length_m, expected_2sigma, rtol=0.05)
        numpy.testing.assert_allclose(metrics.rms_minor_axis_length_m, expected_2sigma, rtol=0.05)
        numpy.testing.assert_allclose(metrics.fwhm_major_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(metrics.fwhm_minor_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(
            metrics.encircled_energy_diameter_m, expected_encircled, rtol=0.05
        )

    def test_uniform_disk_rms_to_fwhm_ratio(self) -> None:
        """For a uniform disk: RMS = R and FWHM = R*sqrt(3), so RMS/FWHM = 1/sqrt(3).
        Contrast with the Gaussian ratio of sqrt(2 ln 2) ~ 1.1774 — the shape of
        the intensity distribution materially changes the RMS-vs-FWHM relationship.
        """
        radius_px = 20.0
        size = 128
        y_idx, x_idx = numpy.mgrid[:size, :size]
        r_px = numpy.hypot(x_idx - (size - 1) / 2.0, y_idx - (size - 1) / 2.0)
        disk = (r_px <= radius_px).astype(numpy.float64)

        metrics = estimate_probe_size(disk, PIXEL_GEOMETRY)

        ratio = metrics.rms_major_axis_length_m / metrics.fwhm_major_axis_length_m
        numpy.testing.assert_allclose(ratio, 1.0 / numpy.sqrt(3.0), rtol=0.05)

    def test_elliptical_disk_round_trip(self) -> None:
        """For an axis-aligned uniform ellipse with semi-axes a >= b: the variance
        along each principal axis is (semi-axis)^2 / 4, so the reported 2-sigma
        length equals the semi-axis length.
        """
        a_px = 30.0  # semi-major along x
        b_px = 15.0  # semi-minor along y
        size = 192
        y_idx, x_idx = numpy.mgrid[:size, :size]
        dx = x_idx - (size - 1) / 2.0
        dy = y_idx - (size - 1) / 2.0
        ellipse = ((dx / a_px) ** 2 + (dy / b_px) ** 2 <= 1.0).astype(numpy.float64)

        metrics = estimate_probe_size(ellipse, PIXEL_GEOMETRY)

        assert metrics.rms_major_axis_length_m > metrics.rms_minor_axis_length_m
        numpy.testing.assert_allclose(metrics.rms_major_axis_length_m, a_px * PIXEL_M, rtol=0.05)
        numpy.testing.assert_allclose(metrics.rms_minor_axis_length_m, b_px * PIXEL_M, rtol=0.05)
        numpy.testing.assert_allclose(metrics.major_axis_tilt_rad, 0.0, atol=0.05)

    def test_zero_signal_returns_all_zeros(self) -> None:
        """When the cleaned image has no power, all metrics fall through to 0.0."""
        metrics = estimate_probe_size(numpy.zeros((64, 64)), PIXEL_GEOMETRY)

        assert metrics == ProbeSizeMetrics(
            major_axis_tilt_rad=0.0,
            minor_axis_tilt_rad=0.0,
            fwhm_major_axis_length_m=0.0,
            fwhm_minor_axis_length_m=0.0,
            rms_major_axis_length_m=0.0,
            rms_minor_axis_length_m=0.0,
            encircled_energy_diameter_m=0.0,
        )

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match='2-dimensional'):
            estimate_probe_size(numpy.zeros((4, 8, 8)), PIXEL_GEOMETRY)

    @pytest.mark.parametrize('bad_fraction', [0.0, -0.1, 1.1, 2.0])
    def test_rejects_invalid_energy_fraction(self, bad_fraction: float) -> None:
        with pytest.raises(ValueError, match='energy_fraction'):
            estimate_probe_size(numpy.ones((16, 16)), PIXEL_GEOMETRY, energy_fraction=bad_fraction)

    def test_rejects_negative_mad_threshold(self) -> None:
        with pytest.raises(ValueError, match='mad_threshold'):
            estimate_probe_size(numpy.ones((16, 16)), PIXEL_GEOMETRY, mad_threshold=-1.0)


class TestComputeShannonEntropy:
    def test_uniform_distribution_has_max_normalized_entropy(self) -> None:
        numpy.testing.assert_allclose(compute_shannon_entropy(numpy.ones((16, 16))), 1.0)

    def test_one_hot_distribution_has_zero_entropy(self) -> None:
        values = numpy.zeros((16, 16))
        values[3, 5] = 1.0
        assert compute_shannon_entropy(values) == 0.0

    def test_empty_mass_returns_zero(self) -> None:
        assert compute_shannon_entropy(numpy.zeros((8, 8))) == 0.0

    def test_negative_values_are_clipped(self) -> None:
        # After clipping, one positive element remains -> zero entropy.
        values = numpy.full((4, 4), -1.0)
        values[0, 0] = 2.0
        assert compute_shannon_entropy(values) == 0.0

    def test_concentrated_has_lower_entropy_than_broad(self) -> None:
        narrow = _gaussian_2d(128, 128, 4.0, 4.0, tilt_rad=0.0)
        broad = _gaussian_2d(128, 128, 32.0, 32.0, tilt_rad=0.0)
        assert compute_shannon_entropy(narrow) < compute_shannon_entropy(broad)

    def test_raw_entropy_of_uniform_equals_log2_n(self) -> None:
        values = numpy.ones((8, 8))
        numpy.testing.assert_allclose(
            compute_shannon_entropy(values, normalize=False), numpy.log2(values.size)
        )


class TestEstimateProbeEntropy:
    def _gaussian_probe(self, sigma_px: float) -> Probe:
        amplitude = numpy.sqrt(_gaussian_2d(128, 128, sigma_px, sigma_px, tilt_rad=0.0))
        return Probe(amplitude.astype(numpy.complex128), PIXEL_GEOMETRY)

    def test_metrics_are_normalized_to_unit_interval(self) -> None:
        metrics = estimate_probe_entropy(self._gaussian_probe(16.0))
        assert 0.0 <= metrics.real_space_intensity_entropy <= 1.0
        assert 0.0 <= metrics.spectral_entropy <= 1.0

    def test_fourier_duality_direction(self) -> None:
        """A tightly focused probe is concentrated in real space (low real-space
        entropy) but spread in frequency (high spectral entropy); a broad probe
        is the reverse."""
        narrow = estimate_probe_entropy(self._gaussian_probe(4.0))
        broad = estimate_probe_entropy(self._gaussian_probe(32.0))

        assert narrow.real_space_intensity_entropy < broad.real_space_intensity_entropy
        assert narrow.spectral_entropy > broad.spectral_entropy
