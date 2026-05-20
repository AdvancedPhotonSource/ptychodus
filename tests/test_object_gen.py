"""Tests for ptychodus.api.object_gen.

Covers:
  - generate_gaussian_random_field_object
  - _generate_simplex_noise and the simplex coordinate helpers
    (_map_simplex_to_cartesian, _map_cartesian_to_simplex,
     _calculate_vertex_noise_contribution)

Key behaviors verified for the Gaussian random field generator:
  - Correct output shape (1, height_px, width_px) and complex dtype
  - No NaN / Inf values
  - Reproducibility with a fixed RNG seed; non-reproducibility with different seeds
  - Zero spatial mean (DC component forced to zero in spectral domain)
  - Non-trivial field (nonzero variance) with both real and imaginary parts
  - Larger correlation_length_px → spatially smoother field
  - Spectral power concentrated near the characteristic frequency
  - Pixel geometry and center preserved from ObjectGeometry; single-layer output

Key behaviors verified for the simplex noise generator:
  - Correct output shape and dtype; no NaN / Inf values
  - Reproducibility with a fixed RNG seed; non-reproducibility with different seeds
  - Near-zero sample mean (gradient symmetry implies E[noise] = 0)
  - Bounded amplitude and approximately symmetric distribution
  - Spatially smoother than white noise
  - Correlation length proportional to grid_scale_px
  - Dominant spatial frequency inversely proportional to grid_scale_px
  - Coordinate-mapping roundtrip consistency (simplex <-> Cartesian)
  - Visual outputs saved to tmp_path for manual inspection

Note on the simplex kernel support
----------------------------------
The kernel is `max(0, 0.5 - d^2/grid_scale_px^2)^4` where `d` is the
pixel-space distance from a vertex.  The fixed threshold 0.5 is identical
to the standard simplex-noise convention (Gustavson 2012), normalised so
that all three vertices of each simplex cell contribute at every interior
point regardless of `grid_scale_px`.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import numpy.testing
import pytest

from ptychodus.api.object import ObjectGeometry
from ptychodus.api.object_gen import (
    _calculate_vertex_noise_contribution,
    _generate_simplex_noise,
    _map_cartesian_to_simplex,
    _map_simplex_to_cartesian,
    generate_gaussian_random_field_object,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 42) -> numpy.random.Generator:
    return numpy.random.default_rng(seed)


def _correlation_length(image: numpy.ndarray) -> float:
    """Estimate 1-D correlation length (pixels) from the row-averaged autocorrelation."""
    row = image.mean(axis=0)
    row = row - row.mean()
    if row.std() == 0.0:
        return 0.0
    n = len(row)
    acf_full = numpy.fft.irfft(numpy.abs(numpy.fft.rfft(row, n=2 * n)) ** 2)[:n]
    acf = acf_full / acf_full[0]
    crossings = numpy.where(acf < numpy.exp(-1.0))[0]
    return float(crossings[0]) if len(crossings) else float(n)


def _radial_power_spectrum(
    image: numpy.ndarray, num_bins: int = 64
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Radially averaged power spectrum; returns (frequencies, power) in cycles/pixel."""
    h, w = image.shape
    power = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(image))) ** 2
    fy = numpy.fft.fftshift(numpy.fft.fftfreq(h))
    fx = numpy.fft.fftshift(numpy.fft.fftfreq(w))
    FX, FY = numpy.meshgrid(fx, fy)  # noqa: N806
    R = numpy.hypot(FX, FY)  # noqa: N806
    r_max = min(fy.max(), fx.max())
    edges = numpy.linspace(0.0, r_max, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    avg_power = numpy.zeros(num_bins)
    for k in range(num_bins):
        mask = (R >= edges[k]) & (R < edges[k + 1])
        if mask.any():
            avg_power[k] = power[mask].mean()
    return centers, avg_power


def _make_geometry(
    width_px: int = 64,
    height_px: int = 64,
    pixel_width_m: float = 1e-9,
    pixel_height_m: float = 1e-9,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
) -> ObjectGeometry:
    return ObjectGeometry(
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
        center_x_m=center_x_m,
        center_y_m=center_y_m,
    )


# ===========================================================================
# generate_gaussian_random_field_object
# ===========================================================================


class TestGrfOutputShapeAndDtype:
    def test_square_shape(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(64, 64), correlation_length_px=8.0
        )
        assert obj.get_array().shape == (1, 64, 64)

    def test_rectangular_wide_shape(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(80, 40), correlation_length_px=8.0
        )
        assert obj.get_array().shape == (1, 40, 80)

    def test_rectangular_tall_shape(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(40, 80), correlation_length_px=8.0
        )
        assert obj.get_array().shape == (1, 80, 40)

    def test_dtype_is_complex(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(), correlation_length_px=8.0
        )
        assert numpy.issubdtype(obj.get_array().dtype, numpy.complexfloating)

    def test_no_nans(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(), correlation_length_px=8.0
        )
        assert not numpy.any(numpy.isnan(obj.get_array()))

    def test_no_infs(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(), correlation_length_px=8.0
        )
        assert not numpy.any(numpy.isinf(obj.get_array()))


class TestGrfReproducibility:
    def test_same_seed_identical_output(self) -> None:
        geom = _make_geometry()
        arr1 = generate_gaussian_random_field_object(
            _rng(7), geom, correlation_length_px=8.0
        ).get_array()
        arr2 = generate_gaussian_random_field_object(
            _rng(7), geom, correlation_length_px=8.0
        ).get_array()
        numpy.testing.assert_array_equal(arr1, arr2)

    def test_different_seeds_different_output(self) -> None:
        geom = _make_geometry()
        arr1 = generate_gaussian_random_field_object(
            _rng(7), geom, correlation_length_px=8.0
        ).get_array()
        arr2 = generate_gaussian_random_field_object(
            _rng(99), geom, correlation_length_px=8.0
        ).get_array()
        assert not numpy.array_equal(arr1, arr2)

    def test_different_correlation_lengths_different_output(self) -> None:
        geom = _make_geometry()
        arr1 = generate_gaussian_random_field_object(
            _rng(0), geom, correlation_length_px=4.0
        ).get_array()
        arr2 = generate_gaussian_random_field_object(
            _rng(0), geom, correlation_length_px=16.0
        ).get_array()
        assert not numpy.allclose(arr1, arr2)


class TestGrfStatisticalProperties:
    def test_nonzero_variance(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(), correlation_length_px=8.0
        )
        arr = obj.get_array()
        assert arr.std() > 0.0, 'Expected non-trivial field (all-zero output)'

    def test_mean_near_zero(self) -> None:
        """DC component is zeroed in Fourier space, so the spatial mean must be zero."""
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(128, 128), correlation_length_px=8.0
        )
        arr = obj.get_array()
        mean_abs = abs(arr.mean())
        std = arr.std()
        assert mean_abs < 0.1 * std, (
            f'Mean magnitude {mean_abs:.6f} should be negligible compared to std {std:.6f}; '
            'DC component was zeroed in Fourier space.'
        )

    def test_field_is_complex_valued(self) -> None:
        """The imaginary part should be nonzero; this is a complex field."""
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(), correlation_length_px=8.0
        )
        arr = obj.get_array()
        assert arr.imag.std() > 0.0, 'Expected nonzero imaginary part'

    def test_real_and_imaginary_parts_both_nonzero(self) -> None:
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(), correlation_length_px=8.0
        )
        arr = obj.get_array()
        assert arr.real.std() > 0.0
        assert arr.imag.std() > 0.0


class TestGrfSpatialProperties:
    def test_larger_correlation_length_gives_longer_correlation(self) -> None:
        """Larger correlation_length_px should produce a spatially smoother field."""
        geom = _make_geometry(256, 256)
        short_corr = 4.0
        long_corr = 32.0
        arr_short = generate_gaussian_random_field_object(
            _rng(1), geom, correlation_length_px=short_corr
        ).get_array()[0]
        arr_long = generate_gaussian_random_field_object(
            _rng(1), geom, correlation_length_px=long_corr
        ).get_array()[0]
        len_short = _correlation_length(arr_short.real)
        len_long = _correlation_length(arr_long.real)
        assert len_long > len_short, (
            f'Expected longer 1/e correlation for correlation_length_px={long_corr} '
            f'({len_long:.1f} px) vs {short_corr} ({len_short:.1f} px).'
        )

    def test_smoothness_increases_with_correlation_length(self) -> None:
        """Larger correlation_length_px → smaller normalized gradient magnitude."""
        geom = _make_geometry(128, 128)
        arr_short = (
            generate_gaussian_random_field_object(_rng(2), geom, correlation_length_px=2.0)
            .get_array()[0]
            .real
        )
        arr_long = (
            generate_gaussian_random_field_object(_rng(2), geom, correlation_length_px=20.0)
            .get_array()[0]
            .real
        )

        def _normalized_grad(a: numpy.ndarray) -> float:
            gy = numpy.diff(a, axis=0)
            gx = numpy.diff(a, axis=1)
            rms = numpy.sqrt(numpy.mean(gy**2) + numpy.mean(gx**2))
            return float(rms / a.std()) if a.std() > 0 else float('inf')

        grad_short = _normalized_grad(arr_short)
        grad_long = _normalized_grad(arr_long)
        assert grad_long < grad_short, (
            f'Normalized gradient should be smaller for longer correlation: '
            f'{grad_long:.3f} vs {grad_short:.3f}.'
        )

    def test_power_concentrated_near_characteristic_frequency(self) -> None:
        """Most spectral power should fall near f ~ 1 / (2π * correlation_length_px)."""
        correlation_length_px = 16.0
        geom = _make_geometry(256, 256)
        arr = (
            generate_gaussian_random_field_object(
                _rng(3), geom, correlation_length_px=correlation_length_px
            )
            .get_array()[0]
            .real
        )

        power = numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(arr))) ** 2
        h, w = arr.shape
        fy = numpy.fft.fftshift(numpy.fft.fftfreq(h))
        fx = numpy.fft.fftshift(numpy.fft.fftfreq(w))
        FX, FY = numpy.meshgrid(fx, fy)  # noqa: N806
        R = numpy.hypot(FX, FY)  # noqa: N806

        # Characteristic frequency: the Gaussian envelope peaks around
        # k/(2π) where k·correlation_length_px ~ 1, i.e. f ~ 1/(2π·L)
        f_char = 1.0 / (2 * numpy.pi * correlation_length_px)
        band_mask = (R >= f_char / 4) & (R <= 4 * f_char)
        power_in_band = power[band_mask].sum()
        total_power = power.sum()
        if total_power > 0:
            fraction = power_in_band / total_power
            assert fraction > 0.3, (
                f'Only {100 * fraction:.1f}% of power in characteristic band; expected ≥ 30%.'
            )


class TestGrfGeometryMetadata:
    def test_pixel_geometry_preserved(self) -> None:
        geom = _make_geometry(pixel_width_m=2e-9, pixel_height_m=3e-9)
        obj = generate_gaussian_random_field_object(_rng(), geom, correlation_length_px=8.0)
        pg = obj.get_pixel_geometry()
        assert pg.width_m == pytest.approx(2e-9)
        assert pg.height_m == pytest.approx(3e-9)

    def test_center_preserved(self) -> None:
        geom = _make_geometry(center_x_m=1e-6, center_y_m=-2e-6)
        obj = generate_gaussian_random_field_object(_rng(), geom, correlation_length_px=8.0)
        center = obj.get_center()
        assert center.coordinate_x_m == pytest.approx(1e-6)
        assert center.coordinate_y_m == pytest.approx(-2e-6)

    def test_single_layer(self) -> None:
        """Output should have exactly one slice (layer) on the first axis."""
        obj = generate_gaussian_random_field_object(
            _rng(), _make_geometry(), correlation_length_px=8.0
        )
        assert obj.get_array().shape[0] == 1


# ===========================================================================
# _generate_simplex_noise and coordinate helpers
# ===========================================================================


class TestSimplexOutputShapeAndDtype:
    def test_square_image(self) -> None:
        noise = _generate_simplex_noise(_rng(), 64, 64, 4.0)
        assert noise.shape == (64, 64)

    def test_rectangular_wide(self) -> None:
        noise = _generate_simplex_noise(_rng(), 80, 40, 4.0)
        assert noise.shape == (40, 80)

    def test_rectangular_tall(self) -> None:
        noise = _generate_simplex_noise(_rng(), 40, 80, 4.0)
        assert noise.shape == (80, 40)

    def test_dtype_is_floating(self) -> None:
        noise = _generate_simplex_noise(_rng(), 32, 32, 4.0)
        assert numpy.issubdtype(noise.dtype, numpy.floating)

    def test_no_nans(self) -> None:
        noise = _generate_simplex_noise(_rng(), 64, 64, 4.0)
        assert not numpy.any(numpy.isnan(noise))

    def test_no_infs(self) -> None:
        noise = _generate_simplex_noise(_rng(), 64, 64, 4.0)
        assert not numpy.any(numpy.isinf(noise))


class TestSimplexReproducibility:
    def test_same_seed_identical_output(self) -> None:
        noise1 = _generate_simplex_noise(_rng(7), 64, 64, 4.0)
        noise2 = _generate_simplex_noise(_rng(7), 64, 64, 4.0)
        numpy.testing.assert_array_equal(noise1, noise2)

    def test_different_seeds_different_output(self) -> None:
        noise1 = _generate_simplex_noise(_rng(7), 64, 64, 4.0)
        noise2 = _generate_simplex_noise(_rng(99), 64, 64, 4.0)
        assert not numpy.array_equal(noise1, noise2)

    def test_changing_grid_scale_changes_output(self) -> None:
        noise1 = _generate_simplex_noise(_rng(0), 64, 64, 4.0)
        noise2 = _generate_simplex_noise(_rng(0), 64, 64, 8.0)
        assert not numpy.allclose(noise1, noise2)


class TestSimplexStatisticalProperties:
    """Gradient directions are drawn uniformly from [0, 2π], so E[noise] = 0
    by symmetry, and amplitude is bounded by the kernel maximum times 70.
    """

    def test_nonzero_variance(self) -> None:
        noise = _generate_simplex_noise(_rng(), 64, 64, 8.0)
        assert noise.std() > 0.0, 'Expected non-trivial noise (all-zero output)'

    def test_mean_near_zero(self) -> None:
        """Sample mean should be small relative to std over a large image."""
        noise = _generate_simplex_noise(_rng(), 256, 256, 8.0)
        # Relative threshold: mean < 30 % of std
        assert abs(noise.mean()) < 0.3 * noise.std(), (
            f'Mean ({noise.mean():.4f}) is large relative to std ({noise.std():.4f}); '
            'expected near-zero mean from symmetric random gradients.'
        )

    def test_values_bounded(self) -> None:
        """Noise values should be bounded for a small grid scale."""
        noise = _generate_simplex_noise(_rng(), 128, 128, 1.0)
        assert noise.max() < 10.0
        assert noise.min() > -10.0

    def test_distribution_roughly_symmetric(self) -> None:
        """Amplitude distribution should be approximately symmetric (low skewness)."""
        noise = _generate_simplex_noise(_rng(1), 256, 256, 8.0)
        centered = noise - noise.mean()
        var = numpy.mean(centered**2)
        if var > 0:
            skewness = float(numpy.mean(centered**3) / var**1.5)
            assert abs(skewness) < 1.0, f'Skewness {skewness:.3f} too large for noise'


class TestSimplexSpatialProperties:
    """Simplex noise should be spatially smooth with a characteristic scale
    set by grid_scale_px.
    """

    def test_smoothness_compared_to_white_noise(self) -> None:
        """Normalized gradient magnitude must be well below the white-noise baseline.

        For i.i.d. white noise: RMS(grad) / std(noise) ≈ sqrt(2*2) ≈ 2.
        For smooth noise with correlation length L:
            RMS(grad) / std(noise) ≈ sqrt(2) / L.
        With grid_scale_px = 8 we expect the ratio to be << 1.
        """
        noise = _generate_simplex_noise(_rng(), 128, 128, 8.0)
        gy = numpy.diff(noise, axis=0)
        gx = numpy.diff(noise, axis=1)
        rms_grad = numpy.sqrt(numpy.mean(gy**2) + numpy.mean(gx**2))
        normalized = rms_grad / noise.std()
        assert normalized < 1.0, (
            f'Normalized gradient magnitude {normalized:.3f} is too large; '
            'expected smoother than white noise (baseline ≈ 2).'
        )

    def test_correlation_length_scales_with_grid_scale(self) -> None:
        """Larger grid_scale_px → longer spatial correlation length."""
        scale_small, scale_large = 4.0, 16.0
        noise_small = _generate_simplex_noise(_rng(3), 256, 256, scale_small)
        noise_large = _generate_simplex_noise(_rng(3), 256, 256, scale_large)
        len_small = _correlation_length(noise_small)
        len_large = _correlation_length(noise_large)
        assert len_large > len_small, (
            f'Expected longer correlation length for scale={scale_large} '
            f'({len_large:.1f} px) vs scale={scale_small} ({len_small:.1f} px).'
        )

    def test_dominant_frequency_decreases_with_grid_scale(self) -> None:
        """Peak spatial frequency should decrease as grid_scale_px increases."""
        scale_small, scale_large = 4.0, 16.0
        freqs_s, power_s = _radial_power_spectrum(
            _generate_simplex_noise(_rng(1), 256, 256, scale_small)
        )
        freqs_l, power_l = _radial_power_spectrum(
            _generate_simplex_noise(_rng(1), 256, 256, scale_large)
        )
        # Skip DC bin (index 0)
        peak_small = freqs_s[1:][numpy.argmax(power_s[1:])]
        peak_large = freqs_l[1:][numpy.argmax(power_l[1:])]
        assert peak_small > peak_large, (
            f'Peak frequency should decrease with larger scale: '
            f'{peak_small:.4f} (scale={scale_small}) vs {peak_large:.4f} (scale={scale_large}).'
        )

    def test_characteristic_band_contains_most_power(self) -> None:
        """Most spectral power should fall near the characteristic scale 1/grid_scale_px."""
        scale = 8.0
        noise = _generate_simplex_noise(_rng(5), 256, 256, scale)
        freqs, power = _radial_power_spectrum(noise)
        f_char = 1.0 / scale
        power_mid = power[(freqs >= f_char / 4) & (freqs <= 4 * f_char)].sum()
        total = power.sum()
        if total > 0:
            assert power_mid / total > 0.3, (
                f'Only {100 * power_mid / total:.1f}% of power in characteristic band '
                f'[{f_char / 4:.4f}, {4 * f_char:.4f}] cyc/px; expected ≥ 30%.'
            )

    def test_correlation_length_is_proportional_to_grid_scale(self) -> None:
        """Correlation length should grow roughly linearly with grid_scale_px."""
        scales = [4.0, 8.0, 16.0]
        lengths = [
            _correlation_length(_generate_simplex_noise(_rng(0), 256, 256, s)) for s in scales
        ]
        # Each doubling of scale should increase the correlation length
        assert lengths[1] > lengths[0], (
            f'Doubling scale from {scales[0]} to {scales[1]} should increase '
            f'correlation length: {lengths[0]:.1f} → {lengths[1]:.1f} px.'
        )
        assert lengths[2] > lengths[1], (
            f'Doubling scale from {scales[1]} to {scales[2]} should increase '
            f'correlation length: {lengths[1]:.1f} → {lengths[2]:.1f} px.'
        )


class TestSimplexCoordinateMappings:
    """_map_simplex_to_cartesian and _map_cartesian_to_simplex must be inverses."""

    def test_simplex_to_cartesian_and_back(self) -> None:
        rng = _rng()
        xx = rng.uniform(0.0, 100.0, (32, 32))
        yy = rng.uniform(0.0, 100.0, (32, 32))
        scale = 10.0
        ii, jj = _map_simplex_to_cartesian(xx, yy, scale)
        xx_rt, yy_rt = _map_cartesian_to_simplex(ii, jj, scale)
        numpy.testing.assert_allclose(xx_rt, xx, rtol=1e-10, atol=1e-12)
        numpy.testing.assert_allclose(yy_rt, yy, rtol=1e-10, atol=1e-12)

    def test_cartesian_to_simplex_and_back(self) -> None:
        rng = _rng()
        ii = rng.uniform(-5.0, 5.0, (32, 32))
        jj = rng.uniform(-5.0, 5.0, (32, 32))
        scale = 7.0
        xx, yy = _map_cartesian_to_simplex(ii, jj, scale)
        ii_rt, jj_rt = _map_simplex_to_cartesian(xx, yy, scale)
        numpy.testing.assert_allclose(ii_rt, ii, rtol=1e-10, atol=1e-12)
        numpy.testing.assert_allclose(jj_rt, jj, rtol=1e-10, atol=1e-12)

    def test_origin_maps_to_origin(self) -> None:
        xx = numpy.zeros((1, 1))
        yy = numpy.zeros((1, 1))
        ii, jj = _map_simplex_to_cartesian(xx, yy, 5.0)
        numpy.testing.assert_allclose(ii, 0.0, atol=1e-12)
        numpy.testing.assert_allclose(jj, 0.0, atol=1e-12)

    def test_simplex_indices_scale_inversely_with_grid_scale(self) -> None:
        """Doubling grid_scale_px should halve the simplex indices ii, jj."""
        xx = numpy.full((4, 4), 10.0)
        yy = numpy.full((4, 4), 6.0)
        ii1, jj1 = _map_simplex_to_cartesian(xx, yy, grid_scale_px=5.0)
        ii2, jj2 = _map_simplex_to_cartesian(xx, yy, grid_scale_px=10.0)
        numpy.testing.assert_allclose(ii2, ii1 / 2.0, rtol=1e-10)
        numpy.testing.assert_allclose(jj2, jj1 / 2.0, rtol=1e-10)

    def test_vertex_contribution_zero_at_vertex(self) -> None:
        """Kernel × gradient dot product is zero at the vertex itself (displacement = 0)."""
        width, height = 16, 16
        yy, xx = numpy.mgrid[:height, :width].astype(float)
        vertex_i = numpy.zeros((height, width), dtype=int)
        vertex_j = numpy.zeros((height, width), dtype=int)
        rng = _rng()
        angle = 2 * numpy.pi * rng.uniform(size=(height, width))
        grad_x = numpy.cos(angle)
        grad_y = numpy.sin(angle)
        # Place a vertex exactly at pixel (0, 0); all other pixels have nonzero displacement
        scale = 4.0
        contrib = _calculate_vertex_noise_contribution(
            xx, yy, vertex_i, vertex_j, grad_x, grad_y, scale
        )
        # At pixel (0,0) the displacement is exactly zero → contribution = 0
        assert contrib[0, 0] == pytest.approx(0.0, abs=1e-12)


class TestSimplexVisualOutput:
    """Outputs saved to pytest's tmp_path for manual inspection."""

    def test_single_noise_image(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Save a grayscale simplex noise image."""
        scale = 16.0
        noise = _generate_simplex_noise(_rng(), 256, 256, scale)
        vmax = float(numpy.abs(noise).max()) or 1.0

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(noise, cmap='gray', vmin=-vmax, vmax=vmax, interpolation='nearest')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'Simplex Noise  scale={scale:.0f} px  std={noise.std():.3f}')
        ax.axis('off')
        fig.tight_layout()
        out = tmp_path / 'simplex_noise_single.png'
        fig.savefig(out, dpi=100)
        plt.close(fig)
        assert out.exists()

    def test_multi_scale_comparison(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Compare simplex noise texture at four grid scales side by side."""
        scales = [4.0, 8.0, 16.0, 32.0]
        fig, axes = plt.subplots(1, len(scales), figsize=(16, 4))
        for ax, scale in zip(axes, scales):
            noise = _generate_simplex_noise(_rng(0), 128, 128, scale)
            vmax = float(numpy.abs(noise).max()) or 1.0
            ax.imshow(noise, cmap='seismic', vmin=-vmax, vmax=vmax, interpolation='nearest')
            ax.set_title(f'scale={scale:.0f} px\nstd={noise.std():.3f}')
            ax.axis('off')
        fig.suptitle('Simplex Noise at Different Grid Scales')
        fig.tight_layout()
        out = tmp_path / 'simplex_noise_scales.png'
        fig.savefig(out, dpi=100)
        plt.close(fig)
        assert out.exists()

    def test_power_spectrum_plot(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Plot radial power spectra for several scales; characteristic frequency marked."""
        scales = [4.0, 8.0, 16.0]
        fig, ax = plt.subplots(figsize=(7, 4))
        for scale in scales:
            noise = _generate_simplex_noise(_rng(0), 256, 256, scale)
            freqs, power = _radial_power_spectrum(noise)
            ax.semilogy(freqs, power + 1e-30, label=f'scale={scale:.0f} px')
            ax.axvline(1.0 / scale, linestyle='--', alpha=0.4, color='gray')
        ax.set_xlabel('Spatial frequency (cycles/pixel)')
        ax.set_ylabel('Mean power (log scale)')
        ax.set_title('Radial Power Spectrum of Simplex Noise\n(dashed lines = 1/scale)')
        ax.legend()
        fig.tight_layout()
        out = tmp_path / 'simplex_noise_spectrum.png'
        fig.savefig(out, dpi=100)
        plt.close(fig)
        assert out.exists()

    def test_autocorrelation_plot(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Plot 1-D autocorrelation functions; verify 1/e crossing scales with grid_scale."""
        scales = [4.0, 8.0, 16.0]
        fig, ax = plt.subplots(figsize=(7, 4))
        crossings: list[tuple[float, float]] = []
        for scale in scales:
            noise = _generate_simplex_noise(_rng(0), 256, 256, scale)
            row = noise.mean(axis=0)
            row = row - row.mean()
            n = len(row)
            acf_full = numpy.fft.irfft(numpy.abs(numpy.fft.rfft(row, n=2 * n)) ** 2)[:n]
            acf = acf_full / acf_full[0] if acf_full[0] > 0 else acf_full
            lags = numpy.arange(n)
            ax.plot(lags[: n // 2], acf[: n // 2], label=f'scale={scale:.0f} px')
            idx = numpy.where(acf < numpy.exp(-1.0))[0]
            crossing = float(idx[0]) if len(idx) else float(n)
            crossings.append((scale, crossing))

        ax.axhline(numpy.exp(-1.0), linestyle=':', color='black', label='1/e')
        ax.set_xlabel('Lag (pixels)')
        ax.set_ylabel('Normalised autocorrelation')
        ax.set_title('1-D Autocorrelation of Simplex Noise')
        ax.set_xlim(0, 80)
        ax.legend()
        fig.tight_layout()
        out = tmp_path / 'simplex_noise_autocorrelation.png'
        fig.savefig(out, dpi=100)
        plt.close(fig)
        assert out.exists()

        # Quantitative check embedded in the visual test:
        # 1/e crossing should increase with scale
        for (s1, c1), (s2, c2) in zip(crossings[:-1], crossings[1:]):
            assert c2 > c1, (
                f'1/e crossing should increase from scale={s1} ({c1:.1f} px) '
                f'to scale={s2} ({c2:.1f} px).'
            )
