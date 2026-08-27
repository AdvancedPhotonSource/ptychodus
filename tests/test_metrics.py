"""Unit tests for ptychodus.api.metrics."""

from __future__ import annotations

from dataclasses import replace

import numpy
import numpy.testing
import pytest

from ptychodus.api.diffraction import BadPixels, DiffractionPatterns
from ptychodus.api.fourier import fourier_shift_2d
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.illumination import compute_illumination_map
from ptychodus.api.metrics import (
    FourierRingCorrelation,
    ReconstructionResiduals,
    compute_fourier_ring_correlation,
    compute_mean_absolute_error,
    compute_normalized_mutual_information,
    compute_object_comparison,
    compute_peak_signal_to_noise_ratio,
    compute_r_factor,
    compute_reconstruction_residuals,
    compute_root_mean_square_error,
    compute_structural_similarity,
)
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructionAmbiguities


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


class TestBitThresholdCurve:
    def test_half_bit_curve_asymptote_at_large_n(self):
        # With huge pixels_per_ring the per-bin threshold tends to
        # HALF_BIT_ASYMPTOTE (~0.172) — same constant guarded by the existing
        # resolution tests above. The 1/sqrt(N) correction at N=1e6 is ~1e-3.
        frc = _make_frc([1.0, 0.5, 0.0], pixels_per_ring=[1, 1_000_000, 1_000_000])
        curve = frc.get_bit_threshold_curve(0.5)
        numpy.testing.assert_allclose(
            curve[1:], TestBitThresholdResolution.HALF_BIT_ASYMPTOTE, atol=2e-3
        )

    def test_one_bit_curve_asymptote_at_large_n(self):
        frc = _make_frc([1.0, 0.5, 0.2], pixels_per_ring=[1, 1_000_000, 1_000_000])
        curve = frc.get_bit_threshold_curve(1.0)
        numpy.testing.assert_allclose(
            curve[1:], TestBitThresholdResolution.ONE_BIT_ASYMPTOTE, atol=2e-3
        )

    def test_zero_pixels_per_ring_gives_nan(self):
        frc = _make_frc([1.0, 1.0, 0.0], pixels_per_ring=[0, 0, 4])
        curve = frc.get_bit_threshold_curve(0.5)
        assert numpy.isnan(curve[0])
        assert numpy.isnan(curve[1])
        assert numpy.isfinite(curve[2])

    def test_output_shape_matches_correlation(self):
        frc = _make_frc([1.0, 0.5, 0.2, 0.0])
        assert frc.get_bit_threshold_curve().shape == frc.correlation.shape


class TestSpectralSignalToNoiseRatio:
    def test_half_correlation_gives_snr_of_two(self):
        # SSNR = 2 * 0.5 / (1 - 0.5) = 2
        frc = _make_frc([0.5, 0.5, 0.5])
        numpy.testing.assert_allclose(frc.get_spectral_signal_to_noise_ratio(), 2.0)

    def test_unit_correlation_gives_positive_infinity(self):
        # FRC == 1 makes the denominator vanish → +inf.
        frc = _make_frc([1.0, 0.5, 0.0])
        ssnr = frc.get_spectral_signal_to_noise_ratio()
        assert numpy.isposinf(ssnr[0])
        assert ssnr[1] == pytest.approx(2.0)
        assert ssnr[2] == pytest.approx(0.0)

    def test_zero_correlation_gives_zero_snr(self):
        frc = _make_frc([0.0, 0.0])
        numpy.testing.assert_allclose(frc.get_spectral_signal_to_noise_ratio(), 0.0)

    def test_negative_correlation_is_clipped_to_zero(self):
        # Anti-correlation from noise is not a physical signal.
        frc = _make_frc([-0.3, -0.1, 0.5])
        ssnr = frc.get_spectral_signal_to_noise_ratio()
        assert ssnr[0] == pytest.approx(0.0)
        assert ssnr[1] == pytest.approx(0.0)
        assert ssnr[2] == pytest.approx(2.0)

    def test_nan_propagates(self):
        frc = _make_frc([1.0, float('nan'), 0.5])
        ssnr = frc.get_spectral_signal_to_noise_ratio()
        assert numpy.isposinf(ssnr[0])
        assert numpy.isnan(ssnr[1])
        assert ssnr[2] == pytest.approx(2.0)

    def test_output_shape_matches_correlation(self):
        frc = _make_frc([1.0, 0.5, 0.2, 0.0])
        assert frc.get_spectral_signal_to_noise_ratio().shape == frc.correlation.shape


class TestAreaUnderCurve:
    def test_constant_unit_frc_normalized_auc_is_one(self):
        # FRC ≡ 1 over freq = [0, 1, 2, 3] → AUC = 3, normalized = 1.
        frc = _make_frc([1.0, 1.0, 1.0, 1.0])
        assert frc.get_area_under_curve() == pytest.approx(1.0)
        assert frc.get_area_under_curve(normalize=False) == pytest.approx(3.0)

    def test_zero_frc_gives_zero_auc(self):
        frc = _make_frc([0.0, 0.0, 0.0])
        assert frc.get_area_under_curve() == pytest.approx(0.0)
        assert frc.get_area_under_curve(normalize=False) == pytest.approx(0.0)

    def test_triangular_ramp_normalized_auc_is_one_half(self):
        # FRC = 1 - f/(N-1) on freq = [0, 1, 2, 3] → AUC = 1.5, normalized = 0.5.
        frc = _make_frc([1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0])
        assert frc.get_area_under_curve() == pytest.approx(0.5)
        assert frc.get_area_under_curve(normalize=False) == pytest.approx(1.5)

    def test_max_frequency_clips_integration_domain(self):
        # Whole curve: freq = [0..4], FRC ≡ 1 → AUC = 4. Clipping to 2.0 → AUC = 2.
        frc = _make_frc([1.0, 1.0, 1.0, 1.0, 1.0])
        assert frc.get_area_under_curve(normalize=False, max_frequency_per_m=2.0) == pytest.approx(
            2.0
        )
        assert frc.get_area_under_curve(max_frequency_per_m=2.0) == pytest.approx(1.0)

    def test_nan_bins_excluded(self):
        # Drop the bin at freq=1 → integrate over freq = [0, 2, 3].
        # Trapezoid on a constant FRC = 1 still gives span = 3, normalized = 1.
        frc = _make_frc([1.0, float('nan'), 1.0, 1.0])
        assert frc.get_area_under_curve(normalize=False) == pytest.approx(3.0)
        assert frc.get_area_under_curve() == pytest.approx(1.0)

    def test_fewer_than_two_finite_points_returns_nan(self):
        frc = _make_frc([1.0, float('nan'), float('nan')])
        assert numpy.isnan(frc.get_area_under_curve())
        assert numpy.isnan(frc.get_area_under_curve(normalize=False))

    def test_zero_span_returns_nan(self):
        # Clipping to a max below the second frequency leaves a single point.
        frc = _make_frc([1.0, 1.0, 1.0])
        assert numpy.isnan(frc.get_area_under_curve(max_frequency_per_m=0.5))


class TestAverageSignalToNoiseRatio:
    def test_constant_half_correlation_averages_to_two(self):
        frc = _make_frc([0.5, 0.5, 0.5])
        assert frc.get_average_signal_to_noise_ratio() == pytest.approx(2.0)

    def test_infinite_ssnr_bins_excluded(self):
        # FRC == 1 at bin 0 → +inf SSNR there; should be dropped from the mean.
        # Remaining bin has FRC = 0.5 → SSNR = 2.
        frc = _make_frc([1.0, 0.5])
        assert frc.get_average_signal_to_noise_ratio() == pytest.approx(2.0)

    def test_all_nan_returns_nan(self):
        frc = _make_frc([float('nan'), float('nan')])
        assert numpy.isnan(frc.get_average_signal_to_noise_ratio())

    def test_all_infinite_returns_nan(self):
        # Every FRC == 1 → every SSNR is +inf → no finite bins.
        frc = _make_frc([1.0, 1.0])
        assert numpy.isnan(frc.get_average_signal_to_noise_ratio())


class TestResolutionAtSignalToNoiseThreshold:
    def test_snr_two_matches_frc_threshold_one_half(self):
        # SSNR = 2 inverts to FRC = 2 / (2 + 2) = 0.5.
        frc = _make_frc([1.0, 0.8, 0.2])
        snr_resolution = frc.get_resolution_m_at_signal_to_noise_threshold(2.0)
        frc_resolution = frc.get_resolution_m(0.5)
        assert snr_resolution == pytest.approx(frc_resolution)

    def test_snr_zero_matches_frc_threshold_zero(self):
        frc = _make_frc([1.0, 0.5, -0.1])
        snr_resolution = frc.get_resolution_m_at_signal_to_noise_threshold(0.0)
        frc_resolution = frc.get_resolution_m(0.0)
        # Both reach the same below-threshold bin.
        assert (numpy.isnan(snr_resolution) and numpy.isnan(frc_resolution)) or (
            snr_resolution == pytest.approx(frc_resolution)
        )

    def test_negative_snr_raises(self):
        frc = _make_frc([1.0, 0.5, 0.0])
        with pytest.raises(ValueError, match='non-negative'):
            frc.get_resolution_m_at_signal_to_noise_threshold(-1.0)

    def test_round_trip_two_noisy_copies(self):
        # On a realistic noisy spectrum, the SNR=2 resolution should land
        # between Nyquist and the full image span — the same window as the
        # half-bit resolution in the existing round-trip test.
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
        resolution = frc.get_resolution_m_at_signal_to_noise_threshold(2.0)

        nyquist_resolution_m = 2.0 * pixel_size_m
        max_resolvable_m = shape[0] * pixel_size_m
        assert numpy.isfinite(resolution)
        assert nyquist_resolution_m < resolution < max_resolvable_m


class TestComputeRootMeanSquareError:
    def test_identical_arrays_give_zero(self) -> None:
        rng = numpy.random.default_rng(0)
        arr = rng.standard_normal((8, 12)) + 1j * rng.standard_normal((8, 12))
        assert compute_root_mean_square_error(arr, arr) == pytest.approx(0.0)

    def test_constant_offset_matches_hand_computed(self) -> None:
        reference = numpy.zeros((4, 4), dtype=numpy.float64)
        test = numpy.full((4, 4), 3.0)
        # |3 - 0| everywhere → RMSE = 3.
        assert compute_root_mean_square_error(reference, test) == pytest.approx(3.0)

    def test_complex_uses_modulus_not_real_part(self) -> None:
        # Pure imaginary difference: real-part-only metric would give 0;
        # the modulus distance gives 2 everywhere → RMSE = 2.
        reference = numpy.zeros((4, 4), dtype=numpy.complex128)
        test = numpy.full((4, 4), 2j, dtype=numpy.complex128)
        assert compute_root_mean_square_error(reference, test) == pytest.approx(2.0)

    def test_raises_on_shape_mismatch(self) -> None:
        a = numpy.zeros((4, 4))
        b = numpy.zeros((4, 5))
        with pytest.raises(ValueError, match='shape'):
            compute_root_mean_square_error(a, b)


class TestComputeMeanAbsoluteError:
    def test_identical_arrays_give_zero(self) -> None:
        rng = numpy.random.default_rng(0)
        arr = rng.standard_normal((8, 12)) + 1j * rng.standard_normal((8, 12))
        assert compute_mean_absolute_error(arr, arr) == pytest.approx(0.0)

    def test_constant_offset_matches_hand_computed(self) -> None:
        reference = numpy.zeros((4, 4), dtype=numpy.float64)
        test = numpy.full((4, 4), -2.5)
        assert compute_mean_absolute_error(reference, test) == pytest.approx(2.5)

    def test_complex_uses_modulus(self) -> None:
        # 3 + 4j → modulus 5 → MAE 5.
        reference = numpy.zeros((4, 4), dtype=numpy.complex128)
        test = numpy.full((4, 4), 3.0 + 4.0j)
        assert compute_mean_absolute_error(reference, test) == pytest.approx(5.0)

    def test_raises_on_shape_mismatch(self) -> None:
        a = numpy.zeros((4, 4))
        b = numpy.zeros((4, 5))
        with pytest.raises(ValueError, match='shape'):
            compute_mean_absolute_error(a, b)


class TestComputeRFactor:
    def test_identical_arrays_give_zero(self) -> None:
        rng = numpy.random.default_rng(0)
        arr = rng.standard_normal((8, 12)) + 1j * rng.standard_normal((8, 12))
        assert compute_r_factor(arr, arr) == pytest.approx(0.0)

    def test_known_value_matches_hand_computed(self) -> None:
        # |ref| = 2 everywhere, |test - ref| = 1 everywhere
        #   → R = (1 * N) / (2 * N) = 0.5
        reference = numpy.full((4, 4), 2.0)
        test = numpy.full((4, 4), 3.0)
        assert compute_r_factor(reference, test) == pytest.approx(0.5)

    def test_complex_uses_modulus(self) -> None:
        # ref = 4 + 0j → |ref| = 4; diff = 3j → |diff| = 3 → R = 3/4 = 0.75
        reference = numpy.full((4, 4), 4.0 + 0.0j)
        test = numpy.full((4, 4), 4.0 + 3.0j)
        assert compute_r_factor(reference, test) == pytest.approx(0.75)

    def test_zero_reference_returns_nan(self) -> None:
        reference = numpy.zeros((4, 4), dtype=numpy.complex128)
        test = numpy.ones((4, 4), dtype=numpy.complex128)
        assert numpy.isnan(compute_r_factor(reference, test))

    def test_raises_on_shape_mismatch(self) -> None:
        a = numpy.zeros((4, 4))
        b = numpy.zeros((4, 5))
        with pytest.raises(ValueError, match='shape'):
            compute_r_factor(a, b)


class TestComputePeakSignalToNoiseRatio:
    def test_identical_arrays_give_positive_infinity(self) -> None:
        rng = numpy.random.default_rng(0)
        arr = rng.standard_normal((16, 16))
        assert numpy.isposinf(compute_peak_signal_to_noise_ratio(arr, arr))

    def test_known_value_matches_hand_computed(self) -> None:
        # reference = 0..15 reshaped, test differs by 1.0 everywhere.
        # MSE = 1.0; data_range inferred = 15 - 0 = 15.
        # PSNR = 20 * log10(15) ≈ 23.5218 dB.
        reference = numpy.arange(16, dtype=numpy.float64).reshape(4, 4)
        test = reference + 1.0
        psnr = compute_peak_signal_to_noise_ratio(reference, test)
        assert psnr == pytest.approx(20.0 * numpy.log10(15.0), rel=1e-9)

    def test_explicit_data_range_overrides_inferred(self) -> None:
        reference = numpy.arange(16, dtype=numpy.float64).reshape(4, 4)
        test = reference + 1.0
        # Same MSE = 1, but force data_range = 1 → PSNR = 0 dB.
        psnr = compute_peak_signal_to_noise_ratio(reference, test, data_range=1.0)
        assert psnr == pytest.approx(0.0, abs=1e-12)

    def test_raises_on_shape_mismatch(self) -> None:
        a = numpy.zeros((4, 4))
        b = numpy.zeros((4, 5))
        with pytest.raises(ValueError, match='shape'):
            compute_peak_signal_to_noise_ratio(a, b)


class TestComputeStructuralSimilarity:
    def test_identical_arrays_give_one(self) -> None:
        rng = numpy.random.default_rng(0)
        arr = rng.standard_normal((32, 32))
        assert compute_structural_similarity(arr, arr) == pytest.approx(1.0)

    def test_uncorrelated_random_arrays_score_low(self) -> None:
        rng = numpy.random.default_rng(1)
        a = rng.standard_normal((64, 64))
        b = rng.standard_normal((64, 64))
        assert compute_structural_similarity(a, b) < 0.3

    def test_explicit_data_range_is_honored(self) -> None:
        # Same inputs, different data_range → SSIM changes (the regularization
        # constants C1 and C2 in scikit-image scale with data_range).
        rng = numpy.random.default_rng(2)
        a = rng.standard_normal((32, 32))
        b = a + 0.05 * rng.standard_normal((32, 32))
        ssim_small = compute_structural_similarity(a, b, data_range=1.0)
        ssim_large = compute_structural_similarity(a, b, data_range=100.0)
        assert ssim_small != pytest.approx(ssim_large)

    def test_raises_on_shape_mismatch(self) -> None:
        a = numpy.zeros((8, 8))
        b = numpy.zeros((8, 9))
        with pytest.raises(ValueError, match='shape'):
            compute_structural_similarity(a, b)


class TestComputeNormalizedMutualInformation:
    def test_identical_arrays_give_two(self) -> None:
        rng = numpy.random.default_rng(0)
        arr = rng.standard_normal((32, 32))
        assert compute_normalized_mutual_information(arr, arr) == pytest.approx(2.0)

    def test_uncorrelated_random_arrays_score_near_one(self) -> None:
        rng = numpy.random.default_rng(1)
        a = rng.standard_normal((128, 128))
        b = rng.standard_normal((128, 128))
        nmi = compute_normalized_mutual_information(a, b)
        assert nmi == pytest.approx(1.0, abs=0.05)

    def test_invariant_to_affine_remapping(self) -> None:
        # NMI's signature property vs SSIM/PSNR: an affine intensity remap of
        # `test` leaves the joint-histogram structure intact (same per-axis
        # bin partition) so the score should be essentially unchanged.
        rng = numpy.random.default_rng(2)
        reference = rng.standard_normal((64, 64))
        test = reference + 0.1 * rng.standard_normal((64, 64))
        nmi_raw = compute_normalized_mutual_information(reference, test)
        nmi_scaled = compute_normalized_mutual_information(reference, 3.0 * test + 5.0)
        assert nmi_scaled == pytest.approx(nmi_raw, rel=1e-6)

    def test_explicit_bins_is_honored(self) -> None:
        rng = numpy.random.default_rng(3)
        a = rng.standard_normal((32, 32))
        b = a + 0.05 * rng.standard_normal((32, 32))
        nmi_few = compute_normalized_mutual_information(a, b, bins=8)
        nmi_many = compute_normalized_mutual_information(a, b, bins=256)
        assert nmi_few != pytest.approx(nmi_many)

    def test_raises_on_shape_mismatch(self) -> None:
        a = numpy.zeros((8, 8))
        b = numpy.zeros((8, 9))
        with pytest.raises(ValueError, match='shape'):
            compute_normalized_mutual_information(a, b)


_PIXEL_M = 1.0e-9
_OBJ_HEIGHT_PX = 32
_OBJ_WIDTH_PX = 40
_PROBE_HEIGHT_PX = 8
_PROBE_WIDTH_PX = 8


def _metadata() -> ProductMetadata:
    return ProductMetadata(
        name='test',
        comments='',
        detector_distance_m=1.0,
        probe_energy_eV=10_000.0,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_kg=0.0,
        tomography_angle_deg=0.0,
    )


def _make_object(
    *,
    num_layers: int = 1,
    seed: int = 0,
    dtype: numpy.dtype = numpy.dtype(numpy.complex128),
    pixel_m: float = _PIXEL_M,
) -> Object:
    rng = numpy.random.default_rng(seed)
    real = rng.standard_normal((num_layers, _OBJ_HEIGHT_PX, _OBJ_WIDTH_PX))
    imag = rng.standard_normal((num_layers, _OBJ_HEIGHT_PX, _OBJ_WIDTH_PX))
    array = (real + 1j * imag).astype(dtype)
    return Object(
        array=array,
        pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
        layer_spacing_m=[1.0e-6] * (num_layers - 1),
    )


def _make_probes() -> ProbeSequence:
    rng = numpy.random.default_rng(1)
    shape = (1, 1, _PROBE_HEIGHT_PX, _PROBE_WIDTH_PX)
    array = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(numpy.complex128)
    return ProbeSequence(
        array=array,
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=_PIXEL_M, height_m=_PIXEL_M),
    )


def _make_positions() -> ProbePositionSequence:
    points = [
        ProbePosition(index=i, coordinate_x_m=x, coordinate_y_m=y)
        for i, (x, y) in enumerate(
            [(0.0, 0.0), (3 * _PIXEL_M, -2 * _PIXEL_M), (-4 * _PIXEL_M, 1 * _PIXEL_M)]
        )
    ]
    return ProbePositionSequence(points)


def _make_product(
    *,
    num_layers: int = 1,
    seed: int = 0,
    dtype: numpy.dtype = numpy.dtype(numpy.complex128),
    pixel_m: float = _PIXEL_M,
) -> Product:
    return Product(
        metadata=_metadata(),
        probe_positions=_make_positions(),
        probes=_make_probes(),
        object_=_make_object(num_layers=num_layers, seed=seed, dtype=dtype, pixel_m=pixel_m),
        losses=[],
    )


def _apply_ambiguity_to_object(obj: Object, ambiguities: ReconstructionAmbiguities) -> Object:
    coords = obj.get_geometry().get_transverse_coordinates()
    ramp = (
        ambiguities.phase_ramp_x_rad_per_m * coords.position_x_m
        + ambiguities.phase_ramp_y_rad_per_m * coords.position_y_m
    )
    factor = ambiguities.object_scale_factor * numpy.exp(1j * (ambiguities.phase_offset_rad + ramp))
    new_array = obj.get_array().copy()
    new_array[0] = (new_array[0] * factor).astype(new_array.dtype)
    return Object(
        array=new_array,
        pixel_geometry=obj.get_pixel_geometry().copy(),
        center=obj.get_center().copy(),
        layer_spacing_m=list(obj.layer_spacing_m),
    )


class TestObjectComparison:
    def test_identical_products_yield_identity_ambiguities(self) -> None:
        product = _make_product()
        comparison = compute_object_comparison(reference=product, test=product)

        assert comparison.ambiguities.object_scale_factor == pytest.approx(1.0, abs=1e-10)
        assert comparison.ambiguities.phase_offset_rad == pytest.approx(0.0, abs=1e-10)
        assert comparison.ambiguities.phase_ramp_x_rad_per_m == pytest.approx(0.0, abs=1e-2)
        assert comparison.ambiguities.phase_ramp_y_rad_per_m == pytest.approx(0.0, abs=1e-2)
        numpy.testing.assert_allclose(
            comparison.test_complex, comparison.reference_complex, rtol=1e-10, atol=1e-12
        )

    def test_pixel_geometry_passed_through(self) -> None:
        product = _make_product()
        comparison = compute_object_comparison(reference=product, test=product)
        assert comparison.pixel_geometry == PixelGeometry(width_m=_PIXEL_M, height_m=_PIXEL_M)

    def test_constant_phase_offset_recovered(self) -> None:
        reference = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=1.0,
            phase_offset_rad=0.4,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )
        perturbed = replace(
            reference, object_=_apply_ambiguity_to_object(reference.object_, applied)
        )

        comparison = compute_object_comparison(reference=reference, test=perturbed)

        assert comparison.ambiguities.phase_offset_rad == pytest.approx(0.4, abs=1e-6)
        numpy.testing.assert_allclose(
            comparison.test_complex,
            reference.object_.get_layers_flattened(),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_scale_factor_recovered(self) -> None:
        reference = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=2.5,
            phase_offset_rad=0.0,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )
        perturbed = replace(
            reference, object_=_apply_ambiguity_to_object(reference.object_, applied)
        )

        comparison = compute_object_comparison(reference=reference, test=perturbed)

        assert comparison.ambiguities.object_scale_factor == pytest.approx(2.5, rel=1e-6)
        numpy.testing.assert_allclose(
            comparison.test_complex,
            reference.object_.get_layers_flattened(),
            rtol=1e-6,
            atol=1e-10,
        )

    def test_phase_ramp_recovered(self) -> None:
        reference = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=1.0,
            phase_offset_rad=0.0,
            phase_ramp_x_rad_per_m=1.0e8,
            phase_ramp_y_rad_per_m=-2.0e8,
        )
        perturbed = replace(
            reference, object_=_apply_ambiguity_to_object(reference.object_, applied)
        )

        comparison = compute_object_comparison(reference=reference, test=perturbed)

        # The estimator works in the object array's intrinsic coordinate frame,
        # so recovered ramps should match the applied ramps to high precision.
        assert comparison.ambiguities.phase_ramp_x_rad_per_m == pytest.approx(1.0e8, rel=1e-6)
        assert comparison.ambiguities.phase_ramp_y_rad_per_m == pytest.approx(-2.0e8, rel=1e-6)
        numpy.testing.assert_allclose(
            comparison.test_complex,
            reference.object_.get_layers_flattened(),
            rtol=1e-6,
            atol=1e-8,
        )

    def test_sub_pixel_shift_recovered_on_interior(self) -> None:
        # Make a smooth band-limited reference so the shifted test object
        # matches the reference content well after re-registration. Pure noise
        # objects don't admit a clean sub-pixel shift recovery test because
        # phase_cross_correlation needs a structure to lock onto.
        rng = numpy.random.default_rng(7)
        ky = numpy.fft.fftfreq(_OBJ_HEIGHT_PX)
        kx = numpy.fft.fftfreq(_OBJ_WIDTH_PX)
        radius = numpy.hypot(ky[:, None], kx[None, :])
        spectrum = rng.standard_normal((_OBJ_HEIGHT_PX, _OBJ_WIDTH_PX)) + 1j * rng.standard_normal(
            (_OBJ_HEIGHT_PX, _OBJ_WIDTH_PX)
        )
        spectrum *= numpy.exp(-((radius / 0.15) ** 2) / 2.0)
        smooth = numpy.fft.ifft2(spectrum).astype(numpy.complex128)

        reference_obj = Object(
            array=smooth,
            pixel_geometry=PixelGeometry(width_m=_PIXEL_M, height_m=_PIXEL_M),
            center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
        )
        reference = Product(
            metadata=_metadata(),
            probe_positions=_make_positions(),
            probes=_make_probes(),
            object_=reference_obj,
            losses=[],
        )

        shift_dx, shift_dy = 0.37, -0.62
        shifted = fourier_shift_2d(reference_obj.get_array(), dx=shift_dx, dy=shift_dy)
        shifted_obj = Object(
            array=shifted,
            pixel_geometry=PixelGeometry(width_m=_PIXEL_M, height_m=_PIXEL_M),
            center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
        )
        test = replace(reference, object_=shifted_obj)

        comparison = compute_object_comparison(reference=reference, test=test)

        # Crop off a few pixels at every edge — the Fourier shift wraps content
        # there and re-aligning back doesn't fully undo the wrap. The remaining
        # residual is dominated by phase_cross_correlation's ~1/upsample_factor
        # quantization, so demand only that the recovered pair agrees to ~1e-3.
        margin = 4
        ref_inner = comparison.reference_complex[margin:-margin, margin:-margin]
        test_inner = comparison.test_complex[margin:-margin, margin:-margin]
        numpy.testing.assert_allclose(test_inner, ref_inner, atol=1e-3)

    def test_dtype_promotion_to_complex128(self) -> None:
        reference = _make_product(dtype=numpy.dtype(numpy.complex64), seed=0)
        test = _make_product(dtype=numpy.dtype(numpy.complex128), seed=0)

        comparison = compute_object_comparison(reference=reference, test=test)

        assert comparison.reference_complex.dtype == numpy.dtype(numpy.complex128)
        assert comparison.test_complex.dtype == numpy.dtype(numpy.complex128)

    def test_shape_mismatch_raises(self) -> None:
        reference = _make_product()
        small_object = Object(
            array=numpy.zeros((_OBJ_HEIGHT_PX, _OBJ_WIDTH_PX // 2), dtype=numpy.complex128),
            pixel_geometry=PixelGeometry(width_m=_PIXEL_M, height_m=_PIXEL_M),
            center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
        )
        test = replace(reference, object_=small_object)

        with pytest.raises(ValueError, match='shape'):
            compute_object_comparison(reference=reference, test=test)

    def test_pixel_geometry_mismatch_raises(self) -> None:
        reference = _make_product()
        test = _make_product(pixel_m=2.0 * _PIXEL_M)

        with pytest.raises(ValueError, match='pixel geometry'):
            compute_object_comparison(reference=reference, test=test)

    def test_amplitude_and_phase_properties(self) -> None:
        product = _make_product()
        comparison = compute_object_comparison(reference=product, test=product)

        numpy.testing.assert_array_equal(
            comparison.reference_amplitude, numpy.absolute(comparison.reference_complex)
        )
        numpy.testing.assert_array_equal(
            comparison.test_amplitude, numpy.absolute(comparison.test_complex)
        )
        numpy.testing.assert_array_equal(
            comparison.reference_phase, numpy.angle(comparison.reference_complex)
        )
        numpy.testing.assert_array_equal(
            comparison.test_phase, numpy.angle(comparison.test_complex)
        )


class TestObjectComparisonMetricsIntegration:
    """End-to-end plumbing check: build a Product pair, prepare an ObjectComparison,
    and feed it through every metric. Catches breakage in the metric / dataclass
    contract without re-testing the math each metric covers in isolation."""

    def test_all_metrics_return_finite_scalars_of_expected_sign(self) -> None:
        reference = _make_product(seed=0)
        # Different seed → slightly different object so metrics aren't degenerate
        # (RMSE > 0, SSIM < 1).
        test = _make_product(seed=42)
        comparison = compute_object_comparison(reference=reference, test=test)

        rmse = compute_root_mean_square_error(comparison.reference_complex, comparison.test_complex)
        mae = compute_mean_absolute_error(comparison.reference_complex, comparison.test_complex)
        r = compute_r_factor(comparison.reference_complex, comparison.test_complex)
        assert numpy.isfinite(rmse) and rmse > 0.0
        assert numpy.isfinite(mae) and mae > 0.0
        assert numpy.isfinite(r) and r > 0.0

        psnr_amp = compute_peak_signal_to_noise_ratio(
            comparison.reference_amplitude, comparison.test_amplitude
        )
        ssim_amp = compute_structural_similarity(
            comparison.reference_amplitude, comparison.test_amplitude
        )
        assert numpy.isfinite(psnr_amp)
        assert -1.0 <= ssim_amp <= 1.0

        psnr_phase = compute_peak_signal_to_noise_ratio(
            comparison.reference_phase, comparison.test_phase
        )
        ssim_phase = compute_structural_similarity(
            comparison.reference_phase, comparison.test_phase
        )
        assert numpy.isfinite(psnr_phase)
        assert -1.0 <= ssim_phase <= 1.0

        nmi_amp = compute_normalized_mutual_information(
            comparison.reference_amplitude, comparison.test_amplitude
        )
        nmi_phase = compute_normalized_mutual_information(
            comparison.reference_phase, comparison.test_phase
        )
        # Studholme NMI is bounded in [1, 2] in theory; allow a small slack on
        # the lower end since histogram noise on small arrays can dip below.
        assert 0.95 <= nmi_amp <= 2.0
        assert 0.95 <= nmi_phase <= 2.0

        frc = compute_fourier_ring_correlation(
            comparison.reference_complex,
            comparison.test_complex,
            pixel_width_m=comparison.pixel_geometry.width_m,
            pixel_height_m=comparison.pixel_geometry.height_m,
        )
        assert isinstance(frc, FourierRingCorrelation)
        assert frc.correlation.shape == frc.spatial_frequency_per_m.shape


def _simulate_measured(product: Product) -> DiffractionPatterns:
    return generate_diffraction_data(product).get_patterns()


def _no_bad_pixels(measured: DiffractionPatterns) -> BadPixels:
    """An all-good detector mask shaped to match one frame of *measured*."""
    return numpy.zeros(measured.shape[1:], dtype=bool)


class TestComputeReconstructionResiduals:
    def test_self_consistent_inputs_yield_zero_residuals(self) -> None:
        product = _make_product()
        measured = _simulate_measured(product)
        bad_pixels = _no_bad_pixels(measured)

        result = compute_reconstruction_residuals(product, measured, bad_pixels)

        assert isinstance(result, ReconstructionResiduals)
        # Finite pixels are the well-defined R-factor region; NaN marks "no data".
        recip_finite = numpy.isfinite(result.reciprocal_space_error_map)
        assert recip_finite.any()
        numpy.testing.assert_allclose(
            result.reciprocal_space_error_map[recip_finite], 0.0, atol=1e-12
        )
        real_finite = numpy.isfinite(result.real_space_error_map)
        assert real_finite.any()
        numpy.testing.assert_allclose(result.real_space_error_map[real_finite], 0.0, atol=1e-12)

    def test_constant_offset_gives_expected_r_factor_reciprocal(self) -> None:
        product = _make_product()
        baseline = _simulate_measured(product)
        offset = 0.25
        measured = baseline + offset
        bad_pixels = _no_bad_pixels(measured)

        result = compute_reconstruction_residuals(product, measured, bad_pixels)

        # Amplitude R-factor per detector pixel:
        #   R_F(q) = Σ_n |√I_meas,n − √I_pred,n| / Σ_n √I_meas,n
        # with predicted == baseline and measured == baseline + offset.
        meas_amp = numpy.sqrt(numpy.maximum(measured, 0.0))
        pred_amp = numpy.sqrt(numpy.maximum(baseline, 0.0))
        numerator = numpy.absolute(meas_amp - pred_amp).sum(axis=0)
        denominator = meas_amp.sum(axis=0)
        expected_reciprocal = numpy.where(denominator > 0.0, numerator / denominator, numpy.nan)
        numpy.testing.assert_allclose(
            result.reciprocal_space_error_map, expected_reciprocal, atol=1e-10, equal_nan=True
        )

    def test_un_illuminated_pixels_are_nan(self) -> None:
        product = _make_product()
        measured = _simulate_measured(product) + 0.5
        bad_pixels = _no_bad_pixels(measured)

        result = compute_reconstruction_residuals(product, measured, bad_pixels)

        # Wherever no probe lands, the splat never touches the canvas; the illumination map
        # has the same zero footprint, so we can use it as the "no-data" mask.
        illumination = compute_illumination_map(product).photon_number
        un_illuminated = illumination == 0.0
        assert un_illuminated.any(), 'test setup expects at least one un-illuminated pixel'
        assert numpy.all(numpy.isnan(result.real_space_error_map[un_illuminated]))

    def test_bad_pixels_are_masked_in_reciprocal_map(self) -> None:
        product = _make_product()
        baseline = _simulate_measured(product)
        offset = 0.5
        measured = baseline + offset
        bad_pixels = _no_bad_pixels(measured)
        bad_pixels[0, 0] = True

        result = compute_reconstruction_residuals(product, measured, bad_pixels)

        assert numpy.isnan(result.reciprocal_space_error_map[0, 0])
        good = ~bad_pixels
        meas_amp = numpy.sqrt(numpy.maximum(measured, 0.0))
        pred_amp = numpy.sqrt(numpy.maximum(baseline, 0.0))
        numerator = numpy.absolute(meas_amp - pred_amp).sum(axis=0)
        denominator = meas_amp.sum(axis=0)
        expected_r = numpy.where(denominator > 0.0, numerator / denominator, numpy.nan)
        numpy.testing.assert_allclose(
            result.reciprocal_space_error_map[good], expected_r[good], atol=1e-10, equal_nan=True
        )

    def test_geometry_passthrough_matches_product(self) -> None:
        product = _make_product()
        measured = _simulate_measured(product)
        bad_pixels = _no_bad_pixels(measured)

        result = compute_reconstruction_residuals(product, measured, bad_pixels)

        object_geometry = product.object_.get_geometry()
        assert result.real_space_error_map.shape == (
            object_geometry.height_px,
            object_geometry.width_px,
        )
        assert result.reciprocal_space_error_map.shape == measured.shape[1:]
        assert result.object_pixel_geometry == object_geometry.get_pixel_geometry()
        assert result.object_center == object_geometry.get_center()

    def test_shape_mismatch_raises(self) -> None:
        product = _make_product()
        measured = _simulate_measured(product)
        bad_pixels = numpy.zeros((measured.shape[1] + 1, measured.shape[2]), dtype=bool)

        with pytest.raises(ValueError, match='shape'):
            compute_reconstruction_residuals(product, measured, bad_pixels)

    def test_real_space_map_is_invariant_to_scan_density(self) -> None:
        # Duplicating every scan position doubles both the amplitude-residual
        # splat and the measured-amplitude splat at every covered pixel, so
        # their ratio — the real-space R_F map — is unchanged. This is the
        # scan-density invariance the normalization is designed to provide.
        product = _make_product()
        baseline = _simulate_measured(product)
        measured = baseline + 0.5
        bad_pixels = _no_bad_pixels(measured)

        positions = list(product.probe_positions)
        doubled_positions = ProbePositionSequence(positions + positions)
        doubled_product = replace(product, probe_positions=doubled_positions)
        doubled_measured = numpy.concatenate([measured, measured], axis=0)

        baseline_result = compute_reconstruction_residuals(product, measured, bad_pixels)
        doubled_result = compute_reconstruction_residuals(
            doubled_product, doubled_measured, bad_pixels
        )

        numpy.testing.assert_allclose(
            doubled_result.real_space_error_map,
            baseline_result.real_space_error_map,
            atol=1e-10,
            equal_nan=True,
        )
        numpy.testing.assert_allclose(
            doubled_result.reciprocal_space_error_map,
            baseline_result.reciprocal_space_error_map,
            atol=1e-10,
            equal_nan=True,
        )
