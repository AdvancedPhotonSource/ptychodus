"""Unit tests for ptychodus.api.metrics."""

from __future__ import annotations

import numpy
import numpy.testing
import pytest

from ptychodus.api.metrics import (
    FourierRingCorrelation,
    compute_fourier_ring_correlation,
)


def _make_frc(
    correlation: list[float],
    *,
    pixels_per_ring: list[int] | None = None,
    freq_step: float = 1.0,
) -> FourierRingCorrelation:
    n = len(correlation)
    return FourierRingCorrelation(
        spatial_frequency_per_m=numpy.arange(n, dtype=float) * freq_step,
        correlation=numpy.asarray(correlation, dtype=float),
        pixels_per_ring=numpy.asarray(
            pixels_per_ring if pixels_per_ring is not None else [1] * n, dtype=numpy.intp
        ),
    )


class TestComputeFourierRingCorrelation:
    def test_identical_arrays_give_unit_correlation(self):
        rng = numpy.random.default_rng(0)
        arr = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))

        frc = compute_fourier_ring_correlation(arr, arr, 1e-6, 1e-6)

        populated = frc.pixels_per_ring > 0
        numpy.testing.assert_allclose(frc.correlation[populated], 1.0, atol=1e-12)

    def test_uncorrelated_random_arrays_have_low_correlation(self):
        rng = numpy.random.default_rng(1)
        a = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        b = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))

        frc = compute_fourier_ring_correlation(a, b, 1e-6, 1e-6)

        away_from_dc = frc.correlation[4:]
        assert numpy.nanmean(numpy.abs(away_from_dc)) < 0.3

    def test_output_lengths_match(self):
        rng = numpy.random.default_rng(2)
        arr = rng.standard_normal((24, 24)).astype(complex)

        frc = compute_fourier_ring_correlation(arr, arr, 1e-6, 1e-6)

        expected_len = min(arr.shape) // 2 + 1
        assert len(frc.spatial_frequency_per_m) == expected_len
        assert len(frc.correlation) == expected_len
        assert len(frc.pixels_per_ring) == expected_len

    def test_non_square_shape_uses_min_dim_for_truncation(self):
        rng = numpy.random.default_rng(3)
        arr = rng.standard_normal((16, 32)).astype(complex)

        frc = compute_fourier_ring_correlation(arr, arr, 1e-6, 1e-6)

        assert len(frc.correlation) == 16 // 2 + 1

    def test_spatial_frequencies_monotonic_and_start_at_zero(self):
        rng = numpy.random.default_rng(4)
        arr = rng.standard_normal((16, 16)).astype(complex)

        frc = compute_fourier_ring_correlation(arr, arr, 1e-6, 1e-6)

        assert frc.spatial_frequency_per_m[0] == 0.0
        assert numpy.all(numpy.diff(frc.spatial_frequency_per_m) >= 0.0)

    def test_pixels_per_ring_positive_below_nyquist(self):
        rng = numpy.random.default_rng(5)
        arr = rng.standard_normal((32, 32)).astype(complex)

        frc = compute_fourier_ring_correlation(arr, arr, 1e-6, 1e-6)

        assert numpy.all(frc.pixels_per_ring > 0)

    def test_raises_on_shape_mismatch(self):
        a = numpy.zeros((8, 8), dtype=complex)
        b = numpy.zeros((8, 16), dtype=complex)

        with pytest.raises(ValueError, match='shape'):
            compute_fourier_ring_correlation(a, b, 1e-6, 1e-6)

    def test_raises_on_non_2d_input(self):
        a = numpy.zeros(8, dtype=complex)
        b = numpy.zeros(8, dtype=complex)

        with pytest.raises(ValueError, match='2D'):
            compute_fourier_ring_correlation(a, b, 1e-6, 1e-6)

        a3 = numpy.zeros((4, 4, 4), dtype=complex)
        with pytest.raises(ValueError, match='2D'):
            compute_fourier_ring_correlation(a3, a3, 1e-6, 1e-6)

    def test_pixel_geometry_affects_frequency_scale(self):
        rng = numpy.random.default_rng(6)
        arr = rng.standard_normal((16, 16)).astype(complex)

        frc_small = compute_fourier_ring_correlation(arr, arr, 1e-6, 1e-6)
        frc_big = compute_fourier_ring_correlation(arr, arr, 2e-6, 2e-6)

        # Doubling pixel size halves the FFT frequency step.
        numpy.testing.assert_allclose(
            frc_big.spatial_frequency_per_m, 0.5 * frc_small.spatial_frequency_per_m
        )


class TestFourierRingCorrelationResolution:
    def test_returns_nan_when_never_crosses(self):
        frc = _make_frc([1.0, 0.9, 0.8])
        assert numpy.isnan(frc.get_resolution_m(0.5))

    def test_returns_nan_when_crosses_at_dc(self):
        # correlation[0] < threshold → first crossing is at DC (freq=0) → nan.
        frc = _make_frc([0.0, 0.8, 0.4])
        assert numpy.isnan(frc.get_resolution_m(0.5))

    def test_no_interpolation_when_previous_bin_only_touches_threshold(self):
        # diff at prev bin is exactly 0 → use freq[first], don't interpolate.
        frc = _make_frc([1.0, 0.5, 0.0])
        # freq = [0, 1, 2] → resolution = 1 / freq[2] = 0.5
        assert frc.get_resolution_m(0.5) == pytest.approx(0.5)

    def test_linear_interpolation_between_bins(self):
        # correlation = [1.0, 0.8, 0.2], threshold = 0.5, freq = [0, 1, 2]
        # diff = [0.5, 0.3, -0.3] → first=2, prev diff=0.3>0, interpolate
        # alpha = 0.3 / (0.3 - (-0.3)) = 0.5 → crossing = 1 + 0.5*(2-1) = 1.5
        # resolution = 1 / 1.5
        frc = _make_frc([1.0, 0.8, 0.2])
        assert frc.get_resolution_m(0.5) == pytest.approx(1.0 / 1.5)

    def test_skips_nan_bins(self):
        # NaN at bin 1 must be ignored; first valid below-threshold bin is 3.
        frc = _make_frc([1.0, float('nan'), 0.8, 0.2])
        # finite[2] is True, diff[2] = 0.3 > 0 → interpolate between 2 and 3.
        # alpha = 0.3 / (0.3 - (-0.3)) = 0.5 → crossing = 2.5
        assert frc.get_resolution_m(0.5) == pytest.approx(1.0 / 2.5)

    def test_skips_nan_bin_when_directly_before_crossing(self):
        # NaN at bin 2 means we can't interpolate across it → use freq[first].
        frc = _make_frc([1.0, 0.8, float('nan'), 0.2])
        # below = [3], finite[2] = False → no interpolation → freq[3] = 3
        assert frc.get_resolution_m(0.5) == pytest.approx(1.0 / 3.0)


class TestBitThresholdResolution:
    HALF_BIT_SIGMA = 0.5 * (numpy.sqrt(2.0) - 1.0)
    HALF_BIT_ASYMPTOTE = HALF_BIT_SIGMA / (HALF_BIT_SIGMA + 1.0)
    ONE_BIT_ASYMPTOTE = 0.5 / 1.5

    def test_half_bit_threshold_at_n1_equals_one(self):
        # DC bin always has 1 pixel; FRC of an array with itself equals 1 there.
        # The threshold curve also evaluates to 1 at N=1, so diff=0 and the
        # code must NOT linearly interpolate a spurious resolution.
        frc = _make_frc(
            [1.0, 0.9, 0.8],
            pixels_per_ring=[1, 10_000, 10_000],
        )
        # 0.9 and 0.8 are both well above the asymptote (~0.172), so resolution
        # should be nan (never crosses).
        assert numpy.isnan(frc.get_resolution_m_at_bit_threshold(0.5))

    def test_half_bit_threshold_asymptote(self):
        # With huge N the per-bin threshold ≈ HALF_BIT_ASYMPTOTE (~0.172).
        # correlation crosses between bin 1 (0.5) and bin 2 (0.0).
        frc = _make_frc(
            [1.0, 0.5, 0.0],
            pixels_per_ring=[1, 10_000, 10_000],
        )
        resolution = frc.get_resolution_m_at_bit_threshold(0.5)
        # crossing should be between freq=1 and freq=2, so resolution in (0.5, 1).
        assert 0.5 < resolution < 1.0

    def test_one_bit_threshold_asymptote(self):
        # 1-bit asymptote = 1/3. correlation crosses between bin 1 (0.5>1/3)
        # and bin 2 (0.2<1/3).
        frc = _make_frc(
            [1.0, 0.5, 0.2],
            pixels_per_ring=[1, 10_000, 10_000],
        )
        resolution = frc.get_resolution_m_at_bit_threshold(1.0)
        assert 0.5 < resolution < 1.0

    def test_returns_nan_when_always_above_threshold(self):
        frc = _make_frc(
            [1.0, 0.99, 0.98],
            pixels_per_ring=[1, 10_000, 10_000],
        )
        assert numpy.isnan(frc.get_resolution_m_at_bit_threshold(0.5))
        assert numpy.isnan(frc.get_resolution_m_at_bit_threshold(1.0))

    def test_handles_zero_pixels_per_ring(self):
        # First two bins have N=0 → threshold is NaN → those bins are skipped.
        # Bin 2 has N=4: half-bit threshold ≈ 0.699, correlation 0.0 < 0.699,
        # so crossing falls at freq[2] (no interp because bin 1 is NaN).
        frc = _make_frc(
            [1.0, 1.0, 0.0],
            pixels_per_ring=[0, 0, 4],
        )
        assert frc.get_resolution_m_at_bit_threshold(0.5) == pytest.approx(0.5)

    def test_round_trip_two_noisy_copies(self):
        # Standard FRC use case: two independent noisy realisations of the same
        # image, where the underlying signal has a smooth low-pass spectrum.
        # The shared low-frequency content dominates near DC (FRC ≈ 1) and the
        # noise dominates near Nyquist (FRC ≈ 0), so the half-bit threshold is
        # crossed somewhere in between.
        rng = numpy.random.default_rng(7)
        shape = (64, 64)
        pixel_size_m = 1e-6
        ky = numpy.fft.fftfreq(shape[0])
        kx = numpy.fft.fftfreq(shape[1])
        radius = numpy.hypot(ky[:, None], kx[None, :])

        spectrum = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        spectrum *= numpy.exp(-((radius / 0.15) ** 2) / 2.0)
        image = numpy.fft.ifft2(spectrum)

        noise_scale = 0.05 * numpy.abs(image).max()
        noisy_a = image + noise_scale * (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )
        noisy_b = image + noise_scale * (
            rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        )

        frc = compute_fourier_ring_correlation(noisy_a, noisy_b, pixel_size_m, pixel_size_m)
        resolution = frc.get_resolution_m_at_bit_threshold(0.5)

        nyquist_resolution_m = 2.0 * pixel_size_m
        max_resolvable_m = shape[0] * pixel_size_m
        assert numpy.isfinite(resolution)
        assert nyquist_resolution_m < resolution < max_resolvable_m
