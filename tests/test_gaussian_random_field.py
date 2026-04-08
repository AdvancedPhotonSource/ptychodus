"""Tests for generate_gaussian_random_field_object in ptychodus.api.object_gen.

Key behaviors verified:
  - Correct output shape (1, height_px, width_px) and complex dtype
  - No NaN / Inf values
  - Reproducibility with a fixed RNG seed
  - Non-reproducibility with different seeds
  - Zero spatial mean (DC component forced to zero in spectral domain)
  - Non-trivial field (nonzero variance)
  - Correlation length effect: larger correlation_length_px → smoother field
  - Dominant spatial frequency inversely proportional to correlation_length_px
  - Preserved pixel geometry and center from ObjectGeometry
  - Rectangular (non-square) grids handled correctly
"""

import numpy
import numpy.testing
import pytest

from ptychodus.api.object import ObjectGeometry
from ptychodus.api.object_gen import generate_gaussian_random_field_object


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 42) -> numpy.random.Generator:
    return numpy.random.default_rng(seed)


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


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


class TestOutputShapeAndDtype:
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


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
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


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


class TestStatisticalProperties:
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


# ---------------------------------------------------------------------------
# Spatial / correlation properties
# ---------------------------------------------------------------------------


class TestSpatialProperties:
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


# ---------------------------------------------------------------------------
# Geometry metadata
# ---------------------------------------------------------------------------


class TestGeometryMetadata:
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
