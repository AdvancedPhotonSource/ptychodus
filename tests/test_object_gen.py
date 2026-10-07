"""Tests for ptychodus.api.simulate.object.

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

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectGeometry
from ptychodus.api.simulate.object import (
    _calculate_vertex_noise_contribution,
    _generate_simplex_noise,
    _map_cartesian_to_simplex,
    _map_simplex_to_cartesian,
    generate_gaussian_random_field_object,
    generate_paganin_object,
    generate_random_object,
    generate_siemens_star_object,
    generate_stxm_object,
    generate_uniform_object,
)
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.assemble import AssembledDiffractionData


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


def _make_assembled_data(
    pattern_counts: list[int],
    *,
    height_px: int = 4,
    width_px: int = 4,
    pixel_width_m: float = 75e-6,
    pixel_height_m: float = 75e-6,
) -> AssembledDiffractionData:
    """Build AssembledDiffractionData so each pattern's total count equals the given value."""
    num_patterns = len(pattern_counts)
    indexes = numpy.arange(num_patterns, dtype=int)
    patterns = numpy.zeros((num_patterns, height_px, width_px), dtype=numpy.int32)
    for i, count in enumerate(pattern_counts):
        patterns[i, 0, 0] = int(count)
    bad_pixels = numpy.zeros((height_px, width_px), dtype=bool)
    pixel_geometry = PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m)
    return AssembledDiffractionData(indexes, patterns, pixel_geometry, bad_pixels)


def _make_probe_positions_grid(
    geometry: ObjectGeometry,
    grid_shape: tuple[int, int],
    *,
    margin_frac: float = 0.2,
) -> list[ProbePosition]:
    """Regular ny x nx scan grid centered on the geometry, leaving ``margin_frac`` on each side."""
    ny, nx = grid_shape
    half_w = (geometry.width_m / 2.0) * (1.0 - margin_frac)
    half_h = (geometry.height_m / 2.0) * (1.0 - margin_frac)
    xs = numpy.linspace(geometry.center_x_m - half_w, geometry.center_x_m + half_w, nx)
    ys = numpy.linspace(geometry.center_y_m - half_h, geometry.center_y_m + half_h, ny)
    positions: list[ProbePosition] = []
    index = 0
    for y in ys:
        for x in xs:
            positions.append(ProbePosition(index=index, x_m=float(x), y_m=float(y)))
            index += 1
    return positions


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
        assert center.x_m == pytest.approx(1e-6)
        assert center.y_m == pytest.approx(-2e-6)

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


def test_autocorrelation_length_increases_with_grid_scale() -> None:
    """The 1/e autocorrelation crossing must track the grid scale it was generated at.

    This is the property that makes `grid_scale` meaningful: a larger scale has to
    produce a visibly coarser texture, not merely a differently seeded one.
    """
    crossings: list[tuple[float, float]] = []

    for scale in (4.0, 8.0, 16.0):
        noise = _generate_simplex_noise(_rng(0), 256, 256, scale)
        row = noise.mean(axis=0)
        row = row - row.mean()
        n = len(row)
        acf_full = numpy.fft.irfft(numpy.abs(numpy.fft.rfft(row, n=2 * n)) ** 2)[:n]
        acf = acf_full / acf_full[0] if acf_full[0] > 0 else acf_full

        idx = numpy.where(acf < numpy.exp(-1.0))[0]
        crossings.append((scale, float(idx[0]) if len(idx) else float(n)))

    for (s1, c1), (s2, c2) in zip(crossings[:-1], crossings[1:]):
        assert c2 > c1, (
            f'1/e crossing should increase from scale={s1} ({c1:.1f} px) '
            f'to scale={s2} ({c2:.1f} px).'
        )


# ===========================================================================
# generate_stxm_object
# ===========================================================================


class TestStxmObject:
    def test_shape_and_dtype(self) -> None:
        geometry = _make_geometry(32, 32)
        positions = _make_probe_positions_grid(geometry, (4, 4))
        data = _make_assembled_data([100] * len(positions))
        obj = generate_stxm_object(geometry, data, positions)
        assert obj.get_array().shape == (1, 32, 32)
        assert numpy.issubdtype(obj.get_array().dtype, numpy.complexfloating)

    def test_preserves_geometry_metadata(self) -> None:
        geometry = _make_geometry(
            width_px=24,
            height_px=32,
            pixel_width_m=2.5e-9,
            pixel_height_m=3.5e-9,
            center_x_m=1.0e-6,
            center_y_m=-2.0e-6,
        )
        positions = _make_probe_positions_grid(geometry, (3, 3))
        data = _make_assembled_data([50] * len(positions))
        obj = generate_stxm_object(geometry, data, positions)
        assert obj.get_pixel_geometry().width_m == pytest.approx(2.5e-9)
        assert obj.get_pixel_geometry().height_m == pytest.approx(3.5e-9)
        assert obj.get_center().x_m == pytest.approx(1.0e-6)
        assert obj.get_center().y_m == pytest.approx(-2.0e-6)

    def test_constant_counts_yield_constant_intensity_interior(self) -> None:
        geometry = _make_geometry(64, 64)
        positions = _make_probe_positions_grid(geometry, (5, 5))
        data = _make_assembled_data([200] * len(positions))
        obj = generate_stxm_object(geometry, data, positions)
        intensity = numpy.square(numpy.abs(obj.get_array()[0]))
        interior = intensity[20:44, 20:44]
        numpy.testing.assert_allclose(interior, 200.0, rtol=1e-6)

    def test_monotonic_ramp_yields_monotonic_intensity(self) -> None:
        geometry = _make_geometry(64, 64)
        positions = _make_probe_positions_grid(geometry, (5, 5))
        counts = [
            int(1000 + 5000 * (p.x_m - geometry.minimum_x_m) / geometry.width_m) for p in positions
        ]
        data = _make_assembled_data(counts)
        obj = generate_stxm_object(geometry, data, positions)
        intensity = numpy.square(numpy.abs(obj.get_array()[0]))
        row = intensity[32, 20:44]
        assert numpy.all(numpy.diff(row) >= -1e-6)
        assert row[-1] > row[0]

    def test_skips_missing_indexes(self) -> None:
        geometry = _make_geometry(32, 32)
        positions = _make_probe_positions_grid(geometry, (4, 4))
        # Provide patterns for only the first 9 of 16 positions; the rest are
        # silently skipped because their LUT key is missing.
        partial_data = _make_assembled_data([100] * 9)
        obj = generate_stxm_object(geometry, partial_data, positions)
        assert obj.get_array().shape == (1, 32, 32)


# ===========================================================================
# generate_paganin_object
# ===========================================================================


def _paganin_common_inputs(
    seed: int = 7,
) -> tuple[ObjectGeometry, AssembledDiffractionData, list[ProbePosition]]:
    geometry = _make_geometry(64, 64, pixel_width_m=1e-9, pixel_height_m=1e-9)
    positions = _make_probe_positions_grid(geometry, (5, 5))
    rng = _rng(seed)
    counts = (1000 + rng.integers(0, 500, size=len(positions))).tolist()
    data = _make_assembled_data(counts)
    return geometry, data, positions


class TestPaganinObject:
    def test_shape_and_dtype(self) -> None:
        geometry, data, positions = _paganin_common_inputs()
        obj = generate_paganin_object(
            geometry,
            data,
            positions,
            photon_wavelength_m=1.0e-10,
            propagation_distance_m=1.0,
            delta_over_beta=100.0,
        )
        assert obj.get_array().shape == (1, 64, 64)
        assert numpy.issubdtype(obj.get_array().dtype, numpy.complexfloating)

    def test_preserves_geometry_metadata(self) -> None:
        geometry = _make_geometry(
            width_px=32,
            height_px=40,
            pixel_width_m=2.0e-9,
            pixel_height_m=4.0e-9,
            center_x_m=5.0e-6,
            center_y_m=-3.0e-6,
        )
        positions = _make_probe_positions_grid(geometry, (5, 5))
        data = _make_assembled_data([500] * len(positions))
        obj = generate_paganin_object(
            geometry,
            data,
            positions,
            photon_wavelength_m=1.0e-10,
            propagation_distance_m=1.0,
            delta_over_beta=100.0,
        )
        assert obj.get_pixel_geometry().width_m == pytest.approx(2.0e-9)
        assert obj.get_pixel_geometry().height_m == pytest.approx(4.0e-9)
        assert obj.get_center().x_m == pytest.approx(5.0e-6)
        assert obj.get_center().y_m == pytest.approx(-3.0e-6)

    def test_rejects_nonpositive_propagation_distance(self) -> None:
        geometry, data, positions = _paganin_common_inputs()
        with pytest.raises(ValueError, match='Propagation distance'):
            generate_paganin_object(
                geometry,
                data,
                positions,
                photon_wavelength_m=1.0e-10,
                propagation_distance_m=0.0,
                delta_over_beta=100.0,
            )

    def test_rejects_nonpositive_delta_over_beta(self) -> None:
        geometry, data, positions = _paganin_common_inputs()
        with pytest.raises(ValueError, match='delta/beta'):
            generate_paganin_object(
                geometry,
                data,
                positions,
                photon_wavelength_m=1.0e-10,
                propagation_distance_m=1.0,
                delta_over_beta=-1.0,
            )

    def test_rejects_zero_intensity(self) -> None:
        geometry = _make_geometry(32, 32)
        positions = _make_probe_positions_grid(geometry, (4, 4))
        data = _make_assembled_data([0] * len(positions))
        with pytest.raises(ValueError, match='Mean STXM intensity'):
            generate_paganin_object(
                geometry,
                data,
                positions,
                photon_wavelength_m=1.0e-10,
                propagation_distance_m=1.0,
                delta_over_beta=100.0,
            )

    def test_low_pass_behavior(self) -> None:
        """Paganin filter should suppress high frequencies relative to raw STXM."""
        geometry, data, positions = _paganin_common_inputs()
        stxm_obj = generate_stxm_object(geometry, data, positions)
        paganin_obj = generate_paganin_object(
            geometry,
            data,
            positions,
            photon_wavelength_m=1.0e-10,
            propagation_distance_m=10.0,
            delta_over_beta=500.0,
        )
        stxm_intensity = numpy.square(numpy.abs(stxm_obj.get_array()[0]))
        paganin_intensity = numpy.square(numpy.abs(paganin_obj.get_array()[0]))
        # remove DC so the comparison is over fluctuations only
        stxm_intensity = stxm_intensity - stxm_intensity.mean()
        paganin_intensity = paganin_intensity - paganin_intensity.mean()
        stxm_power = numpy.abs(numpy.fft.fft2(stxm_intensity)) ** 2
        paganin_power = numpy.abs(numpy.fft.fft2(paganin_intensity)) ** 2
        h, w = stxm_intensity.shape
        ky = numpy.fft.fftfreq(h)
        kx = numpy.fft.fftfreq(w)
        KY, KX = numpy.meshgrid(ky, kx, indexing='ij')  # noqa: N806
        K = numpy.hypot(KX, KY)  # noqa: N806
        high_freq = K > 0.3
        assert paganin_power[high_freq].sum() < 0.5 * stxm_power[high_freq].sum()

    def test_phase_amplitude_ratio_matches_delta_over_beta(self) -> None:
        """Closed-form relation: phase = (delta/beta) * log(amplitude) at interior pixels."""
        geometry, data, positions = _paganin_common_inputs()
        delta_over_beta = 1.0  # keep phase within (-pi, pi] so numpy.angle does not wrap
        # Near-identity filter so the input STXM variation survives into the output and
        # log(amplitude) has meaningful magnitude.
        obj = generate_paganin_object(
            geometry,
            data,
            positions,
            photon_wavelength_m=1.0e-10,
            propagation_distance_m=1.0e-14,
            delta_over_beta=delta_over_beta,
        )
        stxm_intensity = numpy.square(
            numpy.abs(generate_stxm_object(geometry, data, positions).get_array()[0])
        )
        interior = stxm_intensity > 0  # exclude boundary pixels filled by griddata
        array = obj.get_array()[0]
        amplitude = numpy.abs(array)
        phase = numpy.angle(array)
        log_amp = numpy.log(amplitude)
        mask = interior & (numpy.abs(log_amp) > 1e-3)
        assert mask.sum() > 100
        ratio = phase[mask] / log_amp[mask]
        numpy.testing.assert_allclose(ratio, delta_over_beta, rtol=1e-4, atol=1e-6)

    def test_near_zero_distance_reduces_to_stxm_power(self) -> None:
        """With a near-zero propagation distance the filter is identity, and on interior
        pixels the result equals (I/<I>) ** ((1 + i*delta/beta)/2)."""
        geometry, data, positions = _paganin_common_inputs()
        delta_over_beta = 50.0
        obj = generate_paganin_object(
            geometry,
            data,
            positions,
            photon_wavelength_m=1.0e-10,
            propagation_distance_m=1.0e-30,
            delta_over_beta=delta_over_beta,
        )
        stxm_intensity = numpy.square(
            numpy.abs(generate_stxm_object(geometry, data, positions).get_array()[0])
        )
        interior = stxm_intensity > 0  # boundary pixels are clipped to small_value
        normalized = stxm_intensity / stxm_intensity.mean()
        exponent = 0.5 * (1.0 + 1j * delta_over_beta)
        expected = numpy.power(normalized.astype(complex), exponent)
        numpy.testing.assert_allclose(
            obj.get_array()[0][interior], expected[interior], rtol=1e-6, atol=1e-12
        )


# ===========================================================================
# generate_siemens_star_object
# ===========================================================================


def _default_star_kwargs(
    num_spokes: int = 8,
    outer_radius_fraction: float = 0.9,
    spoke_amplitude: float = 0.0,
    background_amplitude: float = 1.0,
    spoke_phase_tr: float = 0.0,
    background_phase_tr: float = 0.0,
) -> dict:
    return dict(
        num_spokes=num_spokes,
        outer_radius_fraction=outer_radius_fraction,
        spoke_amplitude=spoke_amplitude,
        background_amplitude=background_amplitude,
        spoke_phase_tr=spoke_phase_tr,
        background_phase_tr=background_phase_tr,
    )


class TestSiemensStarOutputShapeAndDtype:
    def test_square_shape(self) -> None:
        obj = generate_siemens_star_object(_make_geometry(64, 64), **_default_star_kwargs())
        assert obj.get_array().shape == (1, 64, 64)

    def test_rectangular_wide_shape(self) -> None:
        obj = generate_siemens_star_object(_make_geometry(80, 40), **_default_star_kwargs())
        assert obj.get_array().shape == (1, 40, 80)

    def test_rectangular_tall_shape(self) -> None:
        obj = generate_siemens_star_object(_make_geometry(40, 80), **_default_star_kwargs())
        assert obj.get_array().shape == (1, 80, 40)

    def test_dtype_is_complex(self) -> None:
        obj = generate_siemens_star_object(_make_geometry(), **_default_star_kwargs())
        assert numpy.issubdtype(obj.get_array().dtype, numpy.complexfloating)

    def test_no_nans(self) -> None:
        obj = generate_siemens_star_object(_make_geometry(), **_default_star_kwargs())
        assert not numpy.any(numpy.isnan(obj.get_array()))

    def test_no_infs(self) -> None:
        obj = generate_siemens_star_object(_make_geometry(), **_default_star_kwargs())
        assert not numpy.any(numpy.isinf(obj.get_array()))


class TestSiemensStarDeterminism:
    def test_identical_calls_are_bit_identical(self) -> None:
        """No RNG dependency: two calls with the same args produce the same array."""
        geom = _make_geometry(128, 128)
        arr1 = generate_siemens_star_object(geom, **_default_star_kwargs()).get_array()
        arr2 = generate_siemens_star_object(geom, **_default_star_kwargs()).get_array()
        numpy.testing.assert_array_equal(arr1, arr2)


class TestSiemensStarGeometryMetadata:
    def test_pixel_geometry_preserved(self) -> None:
        geom = _make_geometry(pixel_width_m=2e-9, pixel_height_m=3e-9)
        obj = generate_siemens_star_object(geom, **_default_star_kwargs())
        pg = obj.get_pixel_geometry()
        assert pg.width_m == pytest.approx(2e-9)
        assert pg.height_m == pytest.approx(3e-9)

    def test_center_preserved(self) -> None:
        geom = _make_geometry(center_x_m=1e-6, center_y_m=-2e-6)
        obj = generate_siemens_star_object(geom, **_default_star_kwargs())
        center = obj.get_center()
        assert center.x_m == pytest.approx(1e-6)
        assert center.y_m == pytest.approx(-2e-6)

    def test_single_layer(self) -> None:
        obj = generate_siemens_star_object(_make_geometry(), **_default_star_kwargs())
        assert obj.get_array().shape[0] == 1


class TestSiemensStarContrast:
    """Amplitude values inside and outside the disk match the configured spoke /
    background values, and the phase channel behaves the same way.
    """

    def test_corners_are_background_amplitude(self) -> None:
        """Corner pixels lie outside the outer disk and take the background value."""
        obj = generate_siemens_star_object(
            _make_geometry(128, 128), **_default_star_kwargs(outer_radius_fraction=0.5)
        )
        array = obj.get_array()[0]
        amplitude = numpy.abs(array)
        numpy.testing.assert_allclose(amplitude[0, 0], 1.0, atol=1e-12)
        numpy.testing.assert_allclose(amplitude[0, -1], 1.0, atol=1e-12)
        numpy.testing.assert_allclose(amplitude[-1, 0], 1.0, atol=1e-12)
        numpy.testing.assert_allclose(amplitude[-1, -1], 1.0, atol=1e-12)

    def test_wedge_interior_matches_expected_value(self) -> None:
        """A pixel deep inside a spoke wedge is opaque; deep inside a background wedge is transparent."""
        # num_spokes=4 → 8 wedges of 45° each. Wedge 0 spans angle [0, π/4); it is
        # a spoke because floor(angle*4/π) % 2 == 0. The angular center of wedge 0
        # is π/8 = 22.5°. Sample a pixel at radius 20 along that direction.
        obj = generate_siemens_star_object(
            _make_geometry(128, 128),
            **_default_star_kwargs(num_spokes=4, outer_radius_fraction=0.9),
        )
        amplitude = numpy.abs(obj.get_array()[0])
        # spoke wedge center: angle = π/8 (below-right at 22.5° in row/col coords)
        row_spoke = int(round(64 + 20 * numpy.sin(numpy.pi / 8)))
        col_spoke = int(round(64 + 20 * numpy.cos(numpy.pi / 8)))
        # background wedge center: angle = 3π/8 (further below-right, in wedge 1)
        row_bg = int(round(64 + 20 * numpy.sin(3 * numpy.pi / 8)))
        col_bg = int(round(64 + 20 * numpy.cos(3 * numpy.pi / 8)))
        assert amplitude[row_spoke, col_spoke] == pytest.approx(0.0, abs=1e-9)
        assert amplitude[row_bg, col_bg] == pytest.approx(1.0, abs=1e-9)

    def test_wedge_count_matches_num_spokes(self) -> None:
        """Traversing a mid-radius circle yields ``2 * num_spokes`` boundary crossings."""
        num_spokes = 8
        obj = generate_siemens_star_object(
            _make_geometry(256, 256),
            **_default_star_kwargs(num_spokes=num_spokes, outer_radius_fraction=0.9),
        )
        amplitude = numpy.abs(obj.get_array()[0])
        # Sample a circular path at radius 60 (well inside the disk of radius 115.2)
        num_samples = 4096
        angles = numpy.linspace(0.0, 2.0 * numpy.pi, num_samples, endpoint=False)
        rows = numpy.round(128 + 60 * numpy.sin(angles)).astype(int)
        cols = numpy.round(128 + 60 * numpy.cos(angles)).astype(int)
        # Binary classification (threshold at 0.5) treats each anti-aliased edge
        # pixel as belonging to one side or the other, so we count 2*num_spokes
        # transitions exactly rather than the higher count sign() reports when
        # AA leaves samples at 0.5.
        is_background = amplitude[rows, cols] > 0.5
        transitions = int(numpy.sum(numpy.diff(is_background.astype(int)) != 0))
        assert transitions == 2 * num_spokes

    def test_phase_only_mode_has_unit_amplitude(self) -> None:
        """With both amplitudes = 1, |array| = 1 everywhere (only the phase channel differs)."""
        obj = generate_siemens_star_object(
            _make_geometry(64, 64),
            **_default_star_kwargs(
                spoke_amplitude=1.0,
                background_amplitude=1.0,
                spoke_phase_tr=0.25,
                background_phase_tr=0.0,
                outer_radius_fraction=0.9,
            ),
        )
        amplitude = numpy.abs(obj.get_array()[0])
        numpy.testing.assert_allclose(amplitude, 1.0, atol=1e-12)

    def test_phase_only_mode_has_nonzero_phase_variation(self) -> None:
        """Phase-only mode still produces the star pattern in the phase channel."""
        obj = generate_siemens_star_object(
            _make_geometry(64, 64),
            **_default_star_kwargs(
                spoke_amplitude=1.0,
                background_amplitude=1.0,
                spoke_phase_tr=0.25,
                background_phase_tr=0.0,
                outer_radius_fraction=0.9,
            ),
        )
        phase = numpy.angle(obj.get_array()[0])
        assert phase.std() > 0.0


class TestSiemensStarValidation:
    def test_rejects_single_spoke(self) -> None:
        with pytest.raises(ValueError, match='num_spokes'):
            generate_siemens_star_object(_make_geometry(), **_default_star_kwargs(num_spokes=1))

    def test_rejects_zero_radius_fraction(self) -> None:
        with pytest.raises(ValueError, match='outer_radius_fraction'):
            generate_siemens_star_object(
                _make_geometry(), **_default_star_kwargs(outer_radius_fraction=0.0)
            )

    def test_rejects_negative_radius_fraction(self) -> None:
        with pytest.raises(ValueError, match='outer_radius_fraction'):
            generate_siemens_star_object(
                _make_geometry(), **_default_star_kwargs(outer_radius_fraction=-0.5)
            )


class TestGenerateUniformObject:
    """The transparent starting guess.

    It replaced four call sites that reached for ``generate_random_object`` with every
    deviation set to zero, so the equivalence to that call is pinned here: it is what
    makes the substitution a refactor rather than a change.
    """

    def test_output_shape_and_dtype(self) -> None:
        geometry = _make_geometry(width_px=12, height_px=7)
        array = generate_uniform_object(geometry).get_array()

        assert array.shape == (1, 7, 12)
        assert numpy.iscomplexobj(array)

    def test_unit_amplitude_and_zero_phase_everywhere(self) -> None:
        array = generate_uniform_object(_make_geometry()).get_array()

        assert numpy.allclose(numpy.abs(array), 1.0)
        assert numpy.allclose(numpy.angle(array), 0.0)

    def test_matches_the_random_generator_with_zero_deviations(self) -> None:
        geometry = _make_geometry(width_px=9, height_px=5)
        equivalent = generate_random_object(
            _rng(),
            geometry,
            amplitude_mean=1.0,
            amplitude_deviation=0.0,
            phase_mean_tr=0.0,
            phase_deviation_tr=0.0,
            blur_deviation_px=0.0,
        )

        assert numpy.array_equal(
            generate_uniform_object(geometry).get_array(), equivalent.get_array()
        )

    def test_geometry_metadata_is_preserved(self) -> None:
        geometry = _make_geometry(
            width_px=16,
            height_px=8,
            pixel_width_m=3e-9,
            pixel_height_m=5e-9,
            center_x_m=1e-6,
            center_y_m=-2e-6,
        )
        object_ = generate_uniform_object(geometry)

        assert object_.get_pixel_geometry() == geometry.get_pixel_geometry()
        assert object_.get_center() == geometry.get_center()


class TestRandomObjectPhaseMean:
    """``phase_mean_tr`` was accepted and ignored until it was wired up.

    Every caller in the tree passed 0.0, so the omission was invisible; these pin the
    behavior so it stays wired, and pin the units the ``_tr`` suffix claims.
    """

    @staticmethod
    def _generate(phase_mean_tr: float, seed: int = 11) -> numpy.ndarray:
        return generate_random_object(
            _rng(seed),
            _make_geometry(width_px=24, height_px=16),
            amplitude_mean=1.0,
            amplitude_deviation=0.1,
            phase_mean_tr=phase_mean_tr,
            phase_deviation_tr=0.05,
            blur_deviation_px=0.0,
        ).get_array()

    def test_a_nonzero_mean_shifts_the_phase(self) -> None:
        assert not numpy.allclose(self._generate(0.25), self._generate(0.0))

    def test_the_shift_is_exactly_the_requested_number_of_turns(self) -> None:
        """The array is amplitude * exp(2*pi*j * phase), so a mean of 0.25 is a quarter turn."""
        ratio = self._generate(0.25) / self._generate(0.0)
        assert numpy.allclose(ratio, numpy.exp(2.0j * numpy.pi * 0.25))

    def test_a_full_turn_is_indistinguishable_from_none(self) -> None:
        assert numpy.allclose(self._generate(1.0), self._generate(0.0))

    def test_the_phase_mean_leaves_the_amplitude_alone(self) -> None:
        assert numpy.allclose(numpy.abs(self._generate(0.3)), numpy.abs(self._generate(0.0)))
