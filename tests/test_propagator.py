"""Unit tests for wavefield propagators in ptychodus.api.propagate."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy
import numpy.testing
import pytest
from scipy.fft import ifftshift

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.propagate import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelTransferFunctionPropagator,
    FresnelTransformPropagator,
    PropagatedWavefield,
    PropagatorParameters,
    choose_propagator,
    compute_far_field_pixel_geometry,
    compute_full_aperture_fresnel_number,
    compute_near_field_pixel_geometry,
    compute_far_field_propagation_distance,
    compute_magnification,
    intensity,
    propagate_wavefield,
)
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.probe import ProbeGeometry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(
    propagation_distance_m: float,
    *,
    width_px: int = 32,
    height_px: int = 32,
    wavelength_m: float = 500e-9,
    pixel_width_m: float = 50e-6,
    pixel_height_m: float = 50e-6,
) -> PropagatorParameters:
    return PropagatorParameters(
        wavelength_m=wavelength_m,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
        propagation_distance_m=propagation_distance_m,
    )


def _gaussian_wavefield(params: PropagatorParameters, sigma_px: float = 5.0) -> numpy.ndarray:
    """Smooth, bandlimited Gaussian wavefield centered on the array."""
    YY, XX = params.get_spatial_coordinates()  # noqa: N806
    return numpy.exp(-(numpy.square(XX) + numpy.square(YY)) / (2.0 * sigma_px**2)).astype(complex)


def _total_intensity(wavefield: numpy.ndarray) -> float:
    return float(numpy.sum(intensity(wavefield)))


# ---------------------------------------------------------------------------
# intensity()
# ---------------------------------------------------------------------------


class TestIntensity:
    def test_pure_real(self) -> None:
        wf = numpy.array([[3.0, 4.0]], dtype=complex)
        numpy.testing.assert_array_equal(intensity(wf), [[9.0, 16.0]])

    def test_pure_imaginary(self) -> None:
        wf = numpy.array([[3j, 4j]], dtype=complex)
        numpy.testing.assert_array_equal(intensity(wf), [[9.0, 16.0]])

    def test_complex(self) -> None:
        # |3+4j|^2 = 25
        wf = numpy.array([[3.0 + 4.0j]], dtype=complex)
        numpy.testing.assert_allclose(intensity(wf), [[25.0]])

    def test_zero_input(self) -> None:
        wf = numpy.zeros((4, 4), dtype=complex)
        numpy.testing.assert_array_equal(intensity(wf), numpy.zeros((4, 4)))

    def test_output_dtype_is_float(self) -> None:
        wf = numpy.ones((4, 4), dtype=complex)
        assert intensity(wf).dtype.kind == 'f'

    def test_output_shape_preserved(self) -> None:
        wf = numpy.ones((5, 7), dtype=complex)
        assert intensity(wf).shape == (5, 7)


# ---------------------------------------------------------------------------
# PropagatorParameters
# ---------------------------------------------------------------------------


class TestPropagatorParameters:
    def test_dx(self) -> None:
        params = _make_params(0.1, wavelength_m=500e-9, pixel_width_m=50e-6)
        assert params.dx == pytest.approx(100.0)

    def test_pixel_aspect_ratio_square(self) -> None:
        params = _make_params(0.1, pixel_width_m=50e-6, pixel_height_m=50e-6)
        assert params.pixel_aspect_ratio == pytest.approx(1.0)

    def test_pixel_aspect_ratio_rectangular(self) -> None:
        params = _make_params(0.1, pixel_width_m=50e-6, pixel_height_m=25e-6)
        assert params.pixel_aspect_ratio == pytest.approx(2.0)

    def test_z(self) -> None:
        # z = 1e-3 m / 500e-9 m = 2000
        params = _make_params(1.0e-3, wavelength_m=500e-9)
        assert params.z == pytest.approx(2.0e3)

    def test_pixel_fresnel_number(self) -> None:
        # dx=100, z=0.1/500e-9=2e5  →  Fr = 100²/2e5 = 0.05
        params = _make_params(0.1, wavelength_m=500e-9, pixel_width_m=50e-6)
        assert params.pixel_fresnel_number_x == pytest.approx(0.05)

    def test_pixel_fresnel_number_ignores_the_pixel_height(self) -> None:
        """Width-only is deliberate, not an oversight: the propagators reach the y-axis
        through pixel_aspect_ratio, and folding the height in here would double-count it.

        Without a non-square witness the square default makes every other assertion on
        this property agree with an area-based definition too.
        """
        square = _make_params(0.1, pixel_width_m=50e-6, pixel_height_m=50e-6)
        tall = _make_params(0.1, pixel_width_m=50e-6, pixel_height_m=25e-6)

        assert tall.pixel_aspect_ratio != pytest.approx(square.pixel_aspect_ratio)
        assert tall.pixel_fresnel_number_x == pytest.approx(square.pixel_fresnel_number_x)

    def test_pixel_fresnel_number_is_signed(self) -> None:
        """Negating the distance must negate Fr, not leave it unchanged.

        Fr enters the single-FFT propagators through the pure phases C2 and _B, which
        have to conjugate for the backward branch to invert the forward branch. An
        absolute value there silently breaks that inverse.
        """
        params_pos = _make_params(+0.1)
        params_neg = _make_params(-0.1)
        assert params_neg.pixel_fresnel_number_x == pytest.approx(
            -params_pos.pixel_fresnel_number_x
        )

    def test_get_spatial_coordinates_shape(self) -> None:
        params = _make_params(0.1, width_px=16, height_px=24)
        YY, XX = params.get_spatial_coordinates()  # noqa: N806
        assert YY.shape == (24, 16)
        assert XX.shape == (24, 16)

    def test_get_spatial_coordinates_zero_at_center(self) -> None:
        # For even N=8, center index is N//2 = 4
        params = _make_params(0.1, width_px=8, height_px=8)
        YY, XX = params.get_spatial_coordinates()  # noqa: N806
        assert XX[4, 4] == 0
        assert YY[4, 4] == 0

    def test_get_spatial_coordinates_range(self) -> None:
        params = _make_params(0.1, width_px=8, height_px=8)
        YY, XX = params.get_spatial_coordinates()  # noqa: N806
        assert int(XX.min()) == -4
        assert int(XX.max()) == 3
        assert int(YY.min()) == -4
        assert int(YY.max()) == 3

    def test_get_frequency_coordinates_shape(self) -> None:
        params = _make_params(0.1, width_px=16, height_px=24)
        FY, FX = params.get_frequency_coordinates()  # noqa: N806
        assert FY.shape == (24, 16)
        assert FX.shape == (24, 16)

    def test_get_frequency_coordinates_dc_at_center(self) -> None:
        # For N=32, fftshift places DC at index 16
        params = _make_params(0.1, width_px=32, height_px=32)
        FY, FX = params.get_frequency_coordinates()  # noqa: N806
        assert FX[16, 16] == pytest.approx(0.0)
        assert FY[16, 16] == pytest.approx(0.0)

    def test_get_frequency_coordinates_min_is_minus_half(self) -> None:
        params = _make_params(0.1, width_px=32, height_px=32)
        FY, FX = params.get_frequency_coordinates()  # noqa: N806
        assert FX.min() == pytest.approx(-0.5)
        assert FY.min() == pytest.approx(-0.5)

    def test_get_frequency_coordinates_max_less_than_half(self) -> None:
        params = _make_params(0.1, width_px=32, height_px=32)
        FY, FX = params.get_frequency_coordinates()  # noqa: N806
        assert FX.max() < 0.5
        assert FY.max() < 0.5


# ---------------------------------------------------------------------------
# AngularSpectrumPropagator
# ---------------------------------------------------------------------------


class TestAngularSpectrumPropagator:
    def test_output_shape_preserved(self) -> None:
        params = _make_params(0.01, width_px=16, height_px=24)
        result = AngularSpectrumPropagator(params).propagate(_gaussian_wavefield(params))
        assert result.shape == (24, 16)

    def test_output_is_complex(self) -> None:
        params = _make_params(0.01)
        result = AngularSpectrumPropagator(params).propagate(_gaussian_wavefield(params))
        assert numpy.iscomplexobj(result)

    def test_zero_distance_is_identity(self) -> None:
        """z=0 gives TF=1 everywhere; propagation must be the identity."""
        params = _make_params(0.0)
        wf = _gaussian_wavefield(params)
        result = AngularSpectrumPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(result, wf, atol=1e-12)

    def test_energy_conservation(self) -> None:
        """Bandlimited Gaussian conserves total intensity under AS propagation.

        With dx=100 the propagating-wave cutoff ratio F²/dx² < 5e-5 for all
        grid frequencies, so the transfer function is unitary over the entire
        spectrum and Parseval's theorem guarantees conservation.
        """
        params = _make_params(0.01, width_px=64, height_px=64)
        wf = _gaussian_wavefield(params, sigma_px=8.0)
        result = AngularSpectrumPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(_total_intensity(result), _total_intensity(wf), rtol=1e-6)

    def test_round_trip(self) -> None:
        """Forward then backward ASP recovers the original amplitude exactly."""
        params_fwd = _make_params(+0.01, width_px=64, height_px=64)
        params_bwd = _make_params(-0.01, width_px=64, height_px=64)
        wf = _gaussian_wavefield(params_fwd, sigma_px=8.0)
        propagated = AngularSpectrumPropagator(params_fwd).propagate(wf)
        recovered = AngularSpectrumPropagator(params_bwd).propagate(propagated)
        numpy.testing.assert_allclose(numpy.abs(recovered), numpy.abs(wf), atol=1e-12)

    def test_uniform_wavefield_intensity_unchanged(self) -> None:
        """A plane wave (uniform amplitude) keeps per-pixel intensity = 1."""
        params = _make_params(0.05)
        wf = numpy.ones((params.height_px, params.width_px), dtype=complex)
        result = AngularSpectrumPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(intensity(result), intensity(wf), atol=1e-10)

    def test_transfer_function_zero_for_evanescent_modes(self) -> None:
        """TF must be zero for spatial frequencies beyond the propagating cutoff.

        Using sub-wavelength pixels (pixel=300 nm < lambda=500 nm, dx=0.6)
        forces evanescent modes to exist at the array corners.
        """
        params = _make_params(
            0.01,
            width_px=32,
            height_px=32,
            wavelength_m=500e-9,
            pixel_width_m=300e-9,
            pixel_height_m=300e-9,
        )
        prop = AngularSpectrumPropagator(params)
        FY, FX = params.get_frequency_coordinates()  # noqa: N806
        ar = params.pixel_aspect_ratio
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        evanescent = F2 / numpy.square(params.dx) >= 1
        assert evanescent.any(), 'Test precondition: no evanescent modes found; adjust parameters'
        numpy.testing.assert_array_equal(prop._transfer_function[ifftshift(evanescent)], 0.0)

    def test_transfer_function_unit_magnitude_for_propagating_modes(self) -> None:
        """TF has |TF|=1 for all propagating spatial frequencies."""
        params = _make_params(0.01)
        prop = AngularSpectrumPropagator(params)
        FY, FX = params.get_frequency_coordinates()  # noqa: N806
        ar = params.pixel_aspect_ratio
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        propagating = F2 / numpy.square(params.dx) < 1
        tf_mag = numpy.abs(prop._transfer_function[ifftshift(propagating)])
        numpy.testing.assert_allclose(tf_mag, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# FresnelTransferFunctionPropagator
# ---------------------------------------------------------------------------


class TestFresnelTransferFunctionPropagator:
    def test_output_shape_preserved(self) -> None:
        params = _make_params(0.01, width_px=16, height_px=24)
        result = FresnelTransferFunctionPropagator(params).propagate(_gaussian_wavefield(params))
        assert result.shape == (24, 16)

    def test_output_is_complex(self) -> None:
        params = _make_params(0.01)
        result = FresnelTransferFunctionPropagator(params).propagate(_gaussian_wavefield(params))
        assert numpy.iscomplexobj(result)

    def test_zero_distance_is_identity(self) -> None:
        params = _make_params(0.0)
        wf = _gaussian_wavefield(params)
        result = FresnelTransferFunctionPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(result, wf, atol=1e-12)

    def test_transfer_function_unit_magnitude_everywhere(self) -> None:
        """The paraxial TF exp(-iπF²z/dx²) has |TF|=1 for every mode."""
        params = _make_params(0.01)
        prop = FresnelTransferFunctionPropagator(params)
        numpy.testing.assert_allclose(numpy.abs(prop._transfer_function), 1.0, atol=1e-12)

    def test_energy_conservation_arbitrary_input(self) -> None:
        """Because |TF|=1 everywhere, any input conserves total intensity."""
        params = _make_params(0.01, width_px=64, height_px=64)
        rng = numpy.random.default_rng(0)
        wf = rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        result = FresnelTransferFunctionPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(_total_intensity(result), _total_intensity(wf), rtol=1e-10)

    def test_round_trip(self) -> None:
        """TF_bwd = conj(TF_fwd) so forward+backward recovers the input exactly."""
        params_fwd = _make_params(+0.01, width_px=64, height_px=64)
        params_bwd = _make_params(-0.01, width_px=64, height_px=64)
        wf = _gaussian_wavefield(params_fwd, sigma_px=8.0)
        propagated = FresnelTransferFunctionPropagator(params_fwd).propagate(wf)
        recovered = FresnelTransferFunctionPropagator(params_bwd).propagate(propagated)
        numpy.testing.assert_allclose(recovered, wf, atol=1e-10)

    def test_uniform_wavefield_intensity_unchanged(self) -> None:
        params = _make_params(0.05)
        wf = numpy.ones((params.height_px, params.width_px), dtype=complex)
        result = FresnelTransferFunctionPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(intensity(result), intensity(wf), atol=1e-10)

    def test_agrees_with_angular_spectrum_in_paraxial_regime(self) -> None:
        """FresnelTF and ASP give nearly identical intensity when F²/dx² << 1.

        With dx=100 and a smooth Gaussian (sigma=12 px), the dominant spatial
        frequencies satisfy r = F²/dx² ~ 1e-8, making the O(r²) deviation
        between sqrt(1-r) and (1-r/2) negligible.
        """
        params = _make_params(0.001, width_px=64, height_px=64)
        wf = _gaussian_wavefield(params, sigma_px=12.0)
        result_asp = AngularSpectrumPropagator(params).propagate(wf)
        result_ftf = FresnelTransferFunctionPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(numpy.abs(result_asp), numpy.abs(result_ftf), atol=1e-6)


# ---------------------------------------------------------------------------
# FresnelTransformPropagator
# ---------------------------------------------------------------------------


class TestFresnelTransformPropagator:
    def test_output_shape_preserved(self) -> None:
        params = _make_params(0.1, width_px=16, height_px=24)
        result = FresnelTransformPropagator(params).propagate(_gaussian_wavefield(params))
        assert result.shape == (24, 16)

    def test_output_is_complex(self) -> None:
        params = _make_params(0.1)
        result = FresnelTransformPropagator(params).propagate(_gaussian_wavefield(params))
        assert numpy.iscomplexobj(result)

    def test_is_forward_positive_distance(self) -> None:
        assert FresnelTransformPropagator(_make_params(+0.1))._is_forward is True

    def test_is_forward_negative_distance(self) -> None:
        assert FresnelTransformPropagator(_make_params(-0.1))._is_forward is False

    def test_zero_distance_raises(self) -> None:
        with pytest.raises(ValueError, match='nonzero propagation distance'):
            FresnelTransformPropagator(_make_params(0.0))

    def test_agrees_with_fraunhofer_when_fresnel_number_small(self) -> None:
        """When Fr << 1 the quadratic input phase B = exp(iπ Fr X²) ≈ 1,
        so FresnelTransform reduces to the Fraunhofer propagator.

        Parameters: pixel=1 µm, lambda=500 nm, z=100 m →
        Fr = (1e-6/500e-9)² / (100/500e-9) = 4/2e8 ≈ 2e-8  (<<1)
        Fr * (N/2)² ≈ 2e-8 * 1024 ≈ 2e-5  (<<1, so B ≈ 1 across all pixels).
        """
        params = _make_params(
            100.0,
            width_px=64,
            height_px=64,
            pixel_width_m=1e-6,
            pixel_height_m=1e-6,
        )
        wf = _gaussian_wavefield(params, sigma_px=8.0)
        result_fresnel = FresnelTransformPropagator(params).propagate(wf)
        result_fraunhofer = FraunhoferPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(
            numpy.abs(result_fresnel), numpy.abs(result_fraunhofer), rtol=1e-3
        )


# ---------------------------------------------------------------------------
# FraunhoferPropagator
# ---------------------------------------------------------------------------


class TestFraunhoferPropagator:
    def test_output_shape_preserved(self) -> None:
        params = _make_params(10.0, width_px=16, height_px=24)
        result = FraunhoferPropagator(params).propagate(_gaussian_wavefield(params))
        assert result.shape == (24, 16)

    def test_output_is_complex(self) -> None:
        params = _make_params(10.0)
        result = FraunhoferPropagator(params).propagate(_gaussian_wavefield(params))
        assert numpy.iscomplexobj(result)

    def test_is_forward_positive_distance(self) -> None:
        assert FraunhoferPropagator(_make_params(+10.0))._is_forward is True

    def test_is_forward_negative_distance(self) -> None:
        assert FraunhoferPropagator(_make_params(-10.0))._is_forward is False

    def test_zero_distance_raises(self) -> None:
        with pytest.raises(ValueError, match='nonzero propagation distance'):
            FraunhoferPropagator(_make_params(0.0))

    def test_agrees_with_fresnel_transform_when_fresnel_number_small(self) -> None:
        """Mirror of the FresnelTransform test; both should converge for Fr << 1."""
        params = _make_params(
            100.0,
            width_px=64,
            height_px=64,
            pixel_width_m=1e-6,
            pixel_height_m=1e-6,
        )
        wf = _gaussian_wavefield(params, sigma_px=8.0)
        result_fraunhofer = FraunhoferPropagator(params).propagate(wf)
        result_fresnel = FresnelTransformPropagator(params).propagate(wf)
        numpy.testing.assert_allclose(
            numpy.abs(result_fraunhofer), numpy.abs(result_fresnel), rtol=1e-3
        )


# ---------------------------------------------------------------------------
# PropagatedWavefield (dataclass)
# ---------------------------------------------------------------------------


def _make_result(
    *,
    num_steps: int = 3,
    num_modes: int = 2,
    h: int = 4,
    w: int = 5,
    begin_m: float = 0.0,
    end_m: float = 1.0e-3,
    pixel_width_m: float = 50e-6,
    pixel_height_m: float = 50e-6,
    seed: int = 0,
) -> PropagatedWavefield:
    rng = numpy.random.default_rng(seed)
    wf = (
        rng.standard_normal((num_steps, num_modes, h, w))
        + 1j * rng.standard_normal((num_steps, num_modes, h, w))
    ).astype(complex)
    return PropagatedWavefield(
        wavefield=wf,
        begin_coordinate_m=begin_m,
        end_coordinate_m=end_m,
        pixel_geometry=PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m),
    )


class TestPropagatedProbe:
    def test_shape_properties(self) -> None:
        result = _make_result(num_steps=3, num_modes=2, h=4, w=5)
        assert result.num_steps == 3
        assert result.num_incoherent_modes == 2
        assert result.height_px == 4
        assert result.width_px == 5

    def test_intensity_shape(self) -> None:
        result = _make_result(num_steps=3, num_modes=2, h=4, w=5)
        assert result.intensity.shape == (3, 4, 5)

    def test_intensity_is_real_float(self) -> None:
        assert _make_result().intensity.dtype.kind == 'f'

    def test_intensity_equals_sum_of_squared_magnitudes(self) -> None:
        result = _make_result()
        expected = numpy.sum(numpy.abs(result.wavefield) ** 2, axis=1)
        numpy.testing.assert_allclose(result.intensity, expected)

    def test_intensity_recomputed_consistently(self) -> None:
        """Lazy property must return the same values on repeated access."""
        result = _make_result()
        numpy.testing.assert_array_equal(result.intensity, result.intensity)

    def test_get_xy_intensity_equals_intensity_slice(self) -> None:
        result = _make_result(num_steps=3)
        for step in range(result.num_steps):
            numpy.testing.assert_array_equal(result.get_xy_intensity(step), result.intensity[step])

    def test_get_xy_intensity_out_of_bounds(self) -> None:
        result = _make_result(num_steps=3)
        with pytest.raises(IndexError):
            result.get_xy_intensity(3)

    def test_get_zx_intensity_shape_and_average(self) -> None:
        """Even height_px: returned plane averages the two central rows, then transposes."""
        num_steps, num_modes, h, w = 3, 1, 4, 5
        wf = numpy.zeros((num_steps, num_modes, h, w), dtype=complex)
        # central rows for h=4: (h-1)//2 = 1 and h//2 = 2.
        wf[:, 0, 1, :] = 2.0  # |2|^2 = 4
        wf[:, 0, 2, :] = 4.0  # |4|^2 = 16
        result = PropagatedWavefield(
            wavefield=wf,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-3,
            pixel_geometry=PixelGeometry(width_m=50e-6, height_m=50e-6),
        )

        zx = result.get_zx_intensity()

        assert zx.shape == (w, num_steps)  # transposed
        expected_col = numpy.full(w, (4.0 + 16.0) / 2)  # row-average per step
        for step in range(num_steps):
            numpy.testing.assert_allclose(zx[:, step], expected_col)

    def test_get_zy_intensity_shape_and_average(self) -> None:
        """Even width_px: returned plane averages the two central columns, then transposes."""
        num_steps, num_modes, h, w = 3, 1, 5, 4
        wf = numpy.zeros((num_steps, num_modes, h, w), dtype=complex)
        # central cols for w=4: (w-1)//2 = 1 and w//2 = 2.
        wf[:, 0, :, 1] = 2.0  # |2|^2 = 4
        wf[:, 0, :, 2] = 4.0  # |4|^2 = 16
        result = PropagatedWavefield(
            wavefield=wf,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-3,
            pixel_geometry=PixelGeometry(width_m=50e-6, height_m=50e-6),
        )

        zy = result.get_zy_intensity()

        assert zy.shape == (h, num_steps)  # transposed
        expected_col = numpy.full(h, (4.0 + 16.0) / 2)
        for step in range(num_steps):
            numpy.testing.assert_allclose(zy[:, step], expected_col)

    def test_get_xy_wavefield_equals_wavefield_slice(self) -> None:
        result = _make_result(num_steps=3, num_modes=2)
        for step in range(result.num_steps):
            for mode in range(result.num_incoherent_modes):
                numpy.testing.assert_array_equal(
                    result.get_xy_wavefield(step, mode), result.wavefield[step, mode]
                )

    def test_get_xy_wavefield_is_complex(self) -> None:
        assert _make_result().get_xy_wavefield(0, 0).dtype.kind == 'c'

    def test_get_xy_wavefield_distinguishes_modes(self) -> None:
        """Per-mode access must not collapse the mode axis the way `intensity` does."""
        result = _make_result(num_modes=2)
        mode0 = result.get_xy_wavefield(0, 0)
        mode1 = result.get_xy_wavefield(0, 1)

        assert not numpy.allclose(mode0, mode1)
        numpy.testing.assert_allclose(
            result.get_xy_intensity(0), numpy.abs(mode0) ** 2 + numpy.abs(mode1) ** 2
        )

    def test_get_xy_wavefield_mode_out_of_bounds(self) -> None:
        result = _make_result(num_modes=2)
        with pytest.raises(IndexError):
            result.get_xy_wavefield(0, 2)

    def test_get_zx_wavefield_shape_and_average(self) -> None:
        """Even height_px: the complex cut averages the two central phasors."""
        num_steps, num_modes, h, w = 3, 2, 4, 5
        wf = numpy.zeros((num_steps, num_modes, h, w), dtype=complex)
        # central rows for h=4: (h-1)//2 = 1 and h//2 = 2.
        wf[:, 1, 1, :] = 2.0 + 1.0j
        wf[:, 1, 2, :] = 4.0 - 3.0j
        result = PropagatedWavefield(
            wavefield=wf,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-3,
            pixel_geometry=PixelGeometry(width_m=50e-6, height_m=50e-6),
        )

        zx = result.get_zx_wavefield(1)

        assert zx.shape == (w, num_steps)  # transposed
        assert zx.dtype.kind == 'c'
        expected_col = numpy.full(w, ((2.0 + 1.0j) + (4.0 - 3.0j)) / 2)
        for step in range(num_steps):
            numpy.testing.assert_allclose(zx[:, step], expected_col)

    def test_get_zy_wavefield_shape_and_average(self) -> None:
        """Even width_px: the complex cut averages the two central phasors."""
        num_steps, num_modes, h, w = 3, 2, 5, 4
        wf = numpy.zeros((num_steps, num_modes, h, w), dtype=complex)
        # central cols for w=4: (w-1)//2 = 1 and w//2 = 2.
        wf[:, 1, :, 1] = 2.0 + 1.0j
        wf[:, 1, :, 2] = 4.0 - 3.0j
        result = PropagatedWavefield(
            wavefield=wf,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-3,
            pixel_geometry=PixelGeometry(width_m=50e-6, height_m=50e-6),
        )

        zy = result.get_zy_wavefield(1)

        assert zy.shape == (h, num_steps)  # transposed
        assert zy.dtype.kind == 'c'
        expected_col = numpy.full(h, ((2.0 + 1.0j) + (4.0 - 3.0j)) / 2)
        for step in range(num_steps):
            numpy.testing.assert_allclose(zy[:, step], expected_col)

    def test_z_wavefield_shapes_match_intensities(self) -> None:
        result = _make_result(num_steps=3, num_modes=2, h=4, w=5)
        assert result.get_zx_wavefield(0).shape == result.get_zx_intensity().shape
        assert result.get_zy_wavefield(0).shape == result.get_zy_intensity().shape

    def test_single_mode_z_wavefield_intensity_equals_intensity_plane(self) -> None:
        """With one mode the incoherent sum is that mode, so |cut|^2 and the intensity
        cut agree -- the two averaging orders coincide only in this case."""
        result = _make_result(num_steps=3, num_modes=1, h=5, w=5)  # odd: no averaging
        numpy.testing.assert_allclose(
            numpy.abs(result.get_zx_wavefield(0)) ** 2, result.get_zx_intensity()
        )

    def test_frozen_assignment_raises(self) -> None:
        result = _make_result()
        with pytest.raises(FrozenInstanceError):
            result.wavefield = numpy.zeros_like(result.wavefield)  # type: ignore[misc]

    def test_save_npz_round_trip(self, tmp_path: Path) -> None:
        result = _make_result(begin_m=-2e-3, end_m=5e-3)
        file_path = tmp_path / 'propagated_probe.npz'

        result.save_npz(file_path)
        loaded = numpy.load(file_path, allow_pickle=False)

        assert set(loaded.files) == {
            'wavefield',
            'intensity',
            'begin_coordinate_m',
            'end_coordinate_m',
            'pixel_height_m',
            'pixel_width_m',
        }
        assert numpy.iscomplexobj(loaded['wavefield'])
        assert loaded['intensity'].dtype.kind == 'f'
        numpy.testing.assert_array_equal(loaded['wavefield'], result.wavefield)
        numpy.testing.assert_allclose(loaded['intensity'], result.intensity)
        assert float(loaded['begin_coordinate_m']) == pytest.approx(result.begin_coordinate_m)
        assert float(loaded['end_coordinate_m']) == pytest.approx(result.end_coordinate_m)
        assert float(loaded['pixel_height_m']) == pytest.approx(result.pixel_geometry.height_m)
        assert float(loaded['pixel_width_m']) == pytest.approx(result.pixel_geometry.width_m)


# ---------------------------------------------------------------------------
# propagate_wavefield (factory)
# ---------------------------------------------------------------------------


def _flat_pixel_geometry(pixel_m: float = 50e-6) -> PixelGeometry:
    return PixelGeometry(width_m=pixel_m, height_m=pixel_m)


def _source_wavefield(num_modes: int, h: int, w: int, *, seed: int = 0) -> numpy.ndarray:
    """3-D `(modes, h, w)` complex wavefield used as a factory input."""
    rng = numpy.random.default_rng(seed)
    return (
        rng.standard_normal((num_modes, h, w)) + 1j * rng.standard_normal((num_modes, h, w))
    ).astype(complex)


class TestPropagateProbe:
    def test_returns_propagated_probe(self) -> None:
        wf = _source_wavefield(1, 8, 8)
        result = propagate_wavefield(
            wf,
            pixel_geometry=_flat_pixel_geometry(),
            wavelength_m=500e-9,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-3,
            num_steps=3,
        )
        assert isinstance(result, PropagatedWavefield)

    def test_output_wavefield_shape(self) -> None:
        wf = _source_wavefield(num_modes=2, h=8, w=10)
        result = propagate_wavefield(
            wf,
            pixel_geometry=_flat_pixel_geometry(),
            wavelength_m=500e-9,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-3,
            num_steps=4,
        )
        assert result.wavefield.shape == (4, 2, 8, 10)

    def test_dtype_preserved(self) -> None:
        wf = _source_wavefield(1, 8, 8).astype(numpy.complex64)
        result = propagate_wavefield(
            wf,
            pixel_geometry=_flat_pixel_geometry(),
            wavelength_m=500e-9,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-3,
            num_steps=2,
        )
        assert result.wavefield.dtype == numpy.complex64

    def test_metadata_round_trip(self) -> None:
        pg = PixelGeometry(width_m=70e-6, height_m=40e-6)
        result = propagate_wavefield(
            _source_wavefield(1, 8, 8),
            pixel_geometry=pg,
            wavelength_m=500e-9,
            begin_coordinate_m=-2e-3,
            end_coordinate_m=5e-3,
            num_steps=3,
        )
        assert result.pixel_geometry == pg
        assert result.begin_coordinate_m == pytest.approx(-2e-3)
        assert result.end_coordinate_m == pytest.approx(5e-3)

    def test_num_steps_one(self) -> None:
        wf = _source_wavefield(1, 8, 8)
        result = propagate_wavefield(
            wf,
            pixel_geometry=_flat_pixel_geometry(),
            wavelength_m=500e-9,
            begin_coordinate_m=1e-3,
            end_coordinate_m=1e-3,
            num_steps=1,
        )
        assert result.wavefield.shape == (1, 1, 8, 8)

    def test_zero_distance_identity(self) -> None:
        """begin=end=0 → every step is the identity (matches AngularSpectrumPropagator z=0)."""
        wf = _source_wavefield(2, 8, 8)
        result = propagate_wavefield(
            wf,
            pixel_geometry=_flat_pixel_geometry(),
            wavelength_m=500e-9,
            begin_coordinate_m=0.0,
            end_coordinate_m=0.0,
            num_steps=3,
        )
        for step in range(result.num_steps):
            numpy.testing.assert_allclose(result.wavefield[step], wf, atol=1e-12)

    def test_distance_grid_endpoints(self) -> None:
        """First step uses begin_coordinate_m; last step uses end_coordinate_m."""
        wf = _source_wavefield(1, 16, 16)
        pg = _flat_pixel_geometry()
        result = propagate_wavefield(
            wf,
            pixel_geometry=pg,
            wavelength_m=500e-9,
            begin_coordinate_m=1e-3,
            end_coordinate_m=5e-3,
            num_steps=4,
        )

        params_begin = _make_params(
            1e-3, width_px=16, height_px=16, pixel_width_m=pg.width_m, pixel_height_m=pg.height_m
        )
        params_end = _make_params(
            5e-3, width_px=16, height_px=16, pixel_width_m=pg.width_m, pixel_height_m=pg.height_m
        )
        expected_begin = AngularSpectrumPropagator(params_begin).propagate(wf[0])
        expected_end = AngularSpectrumPropagator(params_end).propagate(wf[0])

        numpy.testing.assert_allclose(result.wavefield[0, 0], expected_begin, atol=1e-12)
        numpy.testing.assert_allclose(result.wavefield[-1, 0], expected_end, atol=1e-12)

    def test_invalid_2d_wavefield_raises(self) -> None:
        with pytest.raises(ValueError, match='3-dimensional'):
            propagate_wavefield(
                numpy.zeros((8, 8), dtype=complex),
                pixel_geometry=_flat_pixel_geometry(),
                wavelength_m=500e-9,
                begin_coordinate_m=0.0,
                end_coordinate_m=1e-3,
                num_steps=2,
            )

    def test_invalid_4d_wavefield_raises(self) -> None:
        with pytest.raises(ValueError, match='3-dimensional'):
            propagate_wavefield(
                numpy.zeros((1, 1, 8, 8), dtype=complex),
                pixel_geometry=_flat_pixel_geometry(),
                wavelength_m=500e-9,
                begin_coordinate_m=0.0,
                end_coordinate_m=1e-3,
                num_steps=2,
            )

    def test_multi_mode_independence(self) -> None:
        """Each incoherent mode is propagated independently of the others."""
        params0 = _make_params(2e-3, width_px=16, height_px=16)
        params1 = _make_params(2e-3, width_px=16, height_px=16)
        mode0 = _gaussian_wavefield(params0, sigma_px=3.0)
        mode1 = _gaussian_wavefield(params1, sigma_px=6.0)
        # use distinct phases to ensure modes are not coincidentally identical
        mode1 = mode1 * numpy.exp(1j * 0.5)
        wf = numpy.stack([mode0, mode1], axis=0)

        result = propagate_wavefield(
            wf,
            pixel_geometry=PixelGeometry(width_m=50e-6, height_m=50e-6),
            wavelength_m=500e-9,
            begin_coordinate_m=2e-3,
            end_coordinate_m=2e-3,
            num_steps=1,
        )

        expected0 = AngularSpectrumPropagator(params0).propagate(mode0)
        expected1 = AngularSpectrumPropagator(params1).propagate(mode1)
        numpy.testing.assert_allclose(result.wavefield[0, 0], expected0, atol=1e-12)
        numpy.testing.assert_allclose(result.wavefield[0, 1], expected1, atol=1e-12)

    # ----- Physical-invariant tests -----
    #
    # These validate the factory's wiring via invariants that hold regardless of
    # regime. Absolute correctness of the underlying propagator is pinned
    # separately, against closed-form references, in TestAngularSpectrumAnalytic.

    def test_energy_conservation_across_steps(self) -> None:
        """A bandlimited multi-mode source conserves total |U|^2 at every
        propagation step, because the AS transfer function is unitary over
        the propagating band (see TestAngularSpectrumPropagator)."""
        params = _make_params(0.0, width_px=64, height_px=64)
        mode0 = _gaussian_wavefield(params, sigma_px=8.0)
        mode1 = _gaussian_wavefield(params, sigma_px=6.0) * numpy.exp(0.7j)
        wf = numpy.stack([mode0, mode1], axis=0)
        source_total = float(numpy.sum(numpy.abs(wf) ** 2))

        result = propagate_wavefield(
            wf,
            pixel_geometry=PixelGeometry(width_m=50e-6, height_m=50e-6),
            wavelength_m=500e-9,
            begin_coordinate_m=0.0,
            end_coordinate_m=1e-2,
            num_steps=5,
        )

        for step in range(result.num_steps):
            step_total = float(numpy.sum(numpy.abs(result.wavefield[step]) ** 2))
            assert step_total == pytest.approx(source_total, rel=1e-6)

    def test_forward_backward_round_trip(self) -> None:
        """Propagating to +z and then back by -z should recover the input
        amplitude (the AS round-trip property; here verified through the
        factory rather than the primitive)."""
        z_m = 1e-2
        params = _make_params(z_m, width_px=64, height_px=64)
        mode = _gaussian_wavefield(params, sigma_px=8.0)
        wf = mode[numpy.newaxis, :, :]
        pg = PixelGeometry(width_m=50e-6, height_m=50e-6)

        forward = propagate_wavefield(
            wf,
            pixel_geometry=pg,
            wavelength_m=500e-9,
            begin_coordinate_m=z_m,
            end_coordinate_m=z_m,
            num_steps=1,
        )
        backward = propagate_wavefield(
            forward.wavefield[0],
            pixel_geometry=pg,
            wavelength_m=500e-9,
            begin_coordinate_m=-z_m,
            end_coordinate_m=-z_m,
            num_steps=1,
        )
        numpy.testing.assert_allclose(numpy.abs(backward.wavefield[0]), numpy.abs(wf), atol=1e-12)


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------
#
# The tests above are self-consistency checks: unitarity, round-trips, shift
# equivariance, and comparisons of one propagator against another. All of them
# are satisfied by a transfer function built in the wrong FFT ordering, because
# that error is equivalent to conjugating by a Nyquist chessboard of unit
# modulus. Only comparison against a closed-form solution pins absolute
# correctness, so that is what the classes below do.

_BEAM_WAVELENGTH_M = 1.2398e-10
"""Illumination wavelength at 10 keV."""
_BEAM_PIXEL_M = 20e-9
_BEAM_WAIST_M = 100e-9
_BEAM_RAYLEIGH_M = numpy.pi * numpy.square(_BEAM_WAIST_M) / _BEAM_WAVELENGTH_M
"""Rayleigh range of the reference beam, about 253 um."""


def _beam_params(
    propagation_distance_m: float, *, width_px: int = 128, height_px: int = 128
) -> PropagatorParameters:
    """Tightly focused 10 keV beam: it diffracts strongly over the distances tested."""
    return PropagatorParameters(
        wavelength_m=_BEAM_WAVELENGTH_M,
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=_BEAM_PIXEL_M,
        pixel_height_m=_BEAM_PIXEL_M,
        propagation_distance_m=propagation_distance_m,
    )


def _physical_coordinates(params: PropagatorParameters) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Spatial coordinates in meters rather than pixels."""
    YY, XX = params.get_spatial_coordinates()  # noqa: N806
    return YY * params.pixel_height_m, XX * params.pixel_width_m


def _gaussian_beam(params: PropagatorParameters, waist_m: float = _BEAM_WAIST_M) -> numpy.ndarray:
    """Waist-plane field of a Gaussian beam, with unit on-axis amplitude."""
    yy_m, xx_m = _physical_coordinates(params)
    return numpy.exp(-(numpy.square(xx_m) + numpy.square(yy_m)) / numpy.square(waist_m)).astype(
        complex
    )


def _measure_waist_m(wavefield: numpy.ndarray, params: PropagatorParameters) -> float:
    """Second-moment beam radius: w = sqrt(2<r^2>) for an ``exp(-2r^2/w^2)`` intensity."""
    yy_m, xx_m = _physical_coordinates(params)
    inten = intensity(wavefield)
    r2_m2 = numpy.sum((numpy.square(xx_m) + numpy.square(yy_m)) * inten) / numpy.sum(inten)
    return float(numpy.sqrt(2.0 * r2_m2))


def _rect_aperture(params: PropagatorParameters, aperture_px: int) -> numpy.ndarray:
    """Centered square aperture spanning exactly ``aperture_px`` samples per axis."""
    YY, XX = params.get_spatial_coordinates()  # noqa: N806
    half_px = (aperture_px - 1) / 2
    return ((numpy.abs(XX) <= half_px) & (numpy.abs(YY) <= half_px)).astype(complex)


def _dirichlet_profile(
    num_px: int, aperture_px: int, coordinate_px: numpy.ndarray
) -> numpy.ndarray:
    """Peak-normalized |DFT| of a centered rect of ``aperture_px`` samples.

    The continuous sinc is not the right reference for a sampled aperture; the
    exact discrete analogue is the Dirichlet kernel.
    """
    t = numpy.pi * coordinate_px / num_px
    sin_t = numpy.sin(t)
    safe_sin_t = numpy.where(sin_t == 0.0, 1.0, sin_t)
    kernel = numpy.where(sin_t == 0.0, 1.0, numpy.sin(aperture_px * t) / (aperture_px * safe_sin_t))
    return numpy.abs(kernel)


class TestAngularSpectrumAnalytic:
    """Gaussian-beam closed-form references for AngularSpectrumPropagator."""

    @pytest.mark.parametrize('z_over_rayleigh', [0.5, 1.0, 2.0])
    def test_gaussian_beam_waist_matches_analytic(self, z_over_rayleigh: float) -> None:
        """The beam expands as w(z) = w0 sqrt(1 + (z/zR)^2)."""
        params = _beam_params(z_over_rayleigh * _BEAM_RAYLEIGH_M)
        propagated = AngularSpectrumPropagator(params).propagate(_gaussian_beam(params))

        expected_m = _BEAM_WAIST_M * numpy.sqrt(1.0 + numpy.square(z_over_rayleigh))
        assert _measure_waist_m(propagated, params) == pytest.approx(expected_m, rel=1e-3)

    @pytest.mark.parametrize('z_over_rayleigh', [0.5, 1.0, 2.0])
    def test_gaussian_beam_peak_amplitude_matches_analytic(self, z_over_rayleigh: float) -> None:
        """On-axis amplitude falls as w0/w(z), conserving power as the beam spreads."""
        params = _beam_params(z_over_rayleigh * _BEAM_RAYLEIGH_M)
        propagated = AngularSpectrumPropagator(params).propagate(_gaussian_beam(params))

        expected = 1.0 / numpy.sqrt(1.0 + numpy.square(z_over_rayleigh))
        on_axis = float(numpy.abs(propagated[params.height_px // 2, params.width_px // 2]))
        assert on_axis == pytest.approx(expected, rel=1e-3)

    @pytest.mark.parametrize('z_over_rayleigh', [0.5, 1.0])
    def test_gaussian_beam_gouy_phase_matches_analytic(self, z_over_rayleigh: float) -> None:
        """Removing the plane-wave piston leaves the Gouy phase -arctan(z/zR)."""
        z_m = z_over_rayleigh * _BEAM_RAYLEIGH_M
        params = _beam_params(z_m)
        propagated = AngularSpectrumPropagator(params).propagate(_gaussian_beam(params))

        piston = numpy.exp(-2j * numpy.pi * z_m / _BEAM_WAVELENGTH_M)
        on_axis = propagated[params.height_px // 2, params.width_px // 2] * piston
        expected_rad = -numpy.arctan(z_over_rayleigh)
        assert float(numpy.angle(on_axis)) == pytest.approx(expected_rad, abs=1e-4)

    @pytest.mark.parametrize(('width_px', 'height_px'), [(128, 128), (128, 96)])
    def test_propagated_beam_stays_centered(self, width_px: int, height_px: int) -> None:
        """A centered beam stays centered. Mismatched FFT ordering translates it instead."""
        params = _beam_params(_BEAM_RAYLEIGH_M, width_px=width_px, height_px=height_px)
        propagated = AngularSpectrumPropagator(params).propagate(_gaussian_beam(params))

        peak = numpy.unravel_index(numpy.argmax(intensity(propagated)), propagated.shape)
        assert peak == (height_px // 2, width_px // 2)


class TestFarFieldAnalytic:
    """Closed-form references for the two far-field propagators."""

    def test_fraunhofer_of_rect_aperture_matches_dirichlet(self) -> None:
        """A square aperture diffracts into the Dirichlet kernel of its own width."""
        num_px = 128
        aperture_px = 15
        params = _make_params(5.0, width_px=num_px, height_px=num_px)
        propagated = FraunhoferPropagator(params).propagate(_rect_aperture(params, aperture_px))

        _, XX = params.get_spatial_coordinates()  # noqa: N806
        profile = numpy.abs(propagated[num_px // 2])
        expected = _dirichlet_profile(num_px, aperture_px, XX[num_px // 2])
        numpy.testing.assert_allclose(profile / profile.max(), expected, atol=1e-12)

    def test_fresnel_transform_approaches_fraunhofer_in_far_field(self) -> None:
        """In the far field the Fresnel transform reduces to Fraunhofer.

        Validity is governed by N²·Fr, not Fr alone: the term Fraunhofer drops is
        _B = exp(iπ·Fr·(X²+Y²)) with X_max = N/2. At z=5 m this geometry has
        N²·Fr ≈ 16, which is *not* the far field — the profiles still differ by 1.1e-02
        there. At z=5 km, N²·Fr ≈ 1.6e-02 and they agree to 1.2e-08.
        """
        num_px = 128
        aperture_px = 15
        params = _make_params(5.0e3, width_px=num_px, height_px=num_px)
        assert abs(params.pixel_fresnel_number_x) * num_px**2 < 1.0, (
            'Test precondition: in the far field'
        )

        aperture = _rect_aperture(params, aperture_px)
        fresnel = numpy.abs(FresnelTransformPropagator(params).propagate(aperture)[num_px // 2])
        fraunhofer = numpy.abs(FraunhoferPropagator(params).propagate(aperture)[num_px // 2])
        numpy.testing.assert_allclose(
            fresnel / fresnel.max(), fraunhofer / fraunhofer.max(), atol=1e-6
        )


class TestPropagatorOperatorAlgebra:
    """Structural invariants. These hold for a mis-shifted transfer function too, so
    they complement the analytic references above rather than replacing them."""

    def test_angular_spectrum_propagation_is_additive(self) -> None:
        """P(z1 + z2) equals P(z2) composed with P(z1)."""
        z1_m = 0.3 * _BEAM_RAYLEIGH_M
        z2_m = 0.7 * _BEAM_RAYLEIGH_M
        source = _gaussian_beam(_beam_params(0.0))

        direct = AngularSpectrumPropagator(_beam_params(z1_m + z2_m)).propagate(source)
        stepwise = AngularSpectrumPropagator(_beam_params(z2_m)).propagate(
            AngularSpectrumPropagator(_beam_params(z1_m)).propagate(source)
        )
        numpy.testing.assert_allclose(stepwise, direct, atol=1e-9)


# ---------------------------------------------------------------------------
# Reciprocal-plane pixel geometry
# ---------------------------------------------------------------------------


class TestComputeFarFieldPropagationDistance:
    """The inverse of the far-field relation, for a format that records the pitch not the distance."""

    def test_matches_closed_form(self) -> None:
        distance_m = compute_far_field_propagation_distance(
            PixelGeometry(width_m=75e-6, height_m=50e-6),
            ImageExtent(width_px=256, height_px=192),
            wavelength_m=1.24e-10,
            conjugate_pixel_width_m=1.0e-8,
        )
        assert distance_m == pytest.approx(1.0e-8 * 256 * 75e-6 / 1.24e-10)

    def test_round_trips_through_the_forward_relation(self) -> None:
        # The property that matters: a distance recovered from a recorded sample pixel
        # size has to reproduce that pixel size, or the object is sampled at a scale
        # nothing else in ptychodus agrees with.
        detector = PixelGeometry(width_m=75e-6, height_m=75e-6)
        extent = ImageExtent(width_px=256, height_px=256)
        conjugate_pixel_width_m = 1.884786e-08

        distance_m = compute_far_field_propagation_distance(
            detector,
            extent,
            wavelength_m=1.549802e-10,
            conjugate_pixel_width_m=conjugate_pixel_width_m,
        )
        forward = compute_far_field_pixel_geometry(
            detector, extent, wavelength_m=1.549802e-10, propagation_distance_m=distance_m
        )

        assert forward.width_m == pytest.approx(conjugate_pixel_width_m)

    def test_recovers_the_velociprobe_operating_point(self) -> None:
        # The numbers the fold_slice preprocessing step recorded for the NXSchool IC_1
        # scan: 8 keV on a 256 x 256 Eiger crop, which was taken at 2.335 m.
        distance_m = compute_far_field_propagation_distance(
            PixelGeometry(width_m=75e-6, height_m=75e-6),
            ImageExtent(width_px=256, height_px=256),
            wavelength_m=1.549802e-10,
            conjugate_pixel_width_m=1.884786e-08,
        )

        assert distance_m == pytest.approx(2.335, rel=1e-5)

    def test_a_zero_wavelength_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            compute_far_field_propagation_distance(
                PixelGeometry(width_m=75e-6, height_m=75e-6),
                ImageExtent(width_px=256, height_px=256),
                wavelength_m=0.0,
                conjugate_pixel_width_m=1.0e-8,
            )


class TestComputeFarFieldPixelGeometry:
    def test_matches_closed_form(self) -> None:
        geometry = compute_far_field_pixel_geometry(
            PixelGeometry(width_m=75e-6, height_m=50e-6),
            ImageExtent(width_px=256, height_px=192),
            wavelength_m=1.24e-10,
            propagation_distance_m=1.0,
        )
        assert geometry.width_m == pytest.approx(1.24e-10 / (256 * 75e-6))
        assert geometry.height_m == pytest.approx(1.24e-10 / (192 * 50e-6))

    def test_is_its_own_inverse(self) -> None:
        source = PixelGeometry(width_m=75e-6, height_m=50e-6)
        extent = ImageExtent(width_px=256, height_px=192)
        kwargs = dict(wavelength_m=1.24e-10, propagation_distance_m=1.0)
        once = compute_far_field_pixel_geometry(source, extent, **kwargs)  # type: ignore[arg-type]
        twice = compute_far_field_pixel_geometry(once, extent, **kwargs)  # type: ignore[arg-type]
        assert twice.width_m == pytest.approx(source.width_m)
        assert twice.height_m == pytest.approx(source.height_m)

    def test_uses_absolute_distance(self) -> None:
        source = PixelGeometry(width_m=75e-6, height_m=50e-6)
        extent = ImageExtent(width_px=64, height_px=64)
        forward = compute_far_field_pixel_geometry(
            source, extent, wavelength_m=1e-10, propagation_distance_m=+2.0
        )
        backward = compute_far_field_pixel_geometry(
            source, extent, wavelength_m=1e-10, propagation_distance_m=-2.0
        )
        assert backward == forward

    def test_agrees_with_probe_geometry_from_far_field(self) -> None:
        detector = PixelGeometry(width_m=75e-6, height_m=50e-6)
        extent = ImageExtent(width_px=256, height_px=192)
        expected = ProbeGeometry.from_far_field(
            detector, extent, wavelength_m=1.24e-10, distance_m=1.0
        )
        geometry = compute_far_field_pixel_geometry(
            detector, extent, wavelength_m=1.24e-10, propagation_distance_m=1.0
        )
        assert geometry.width_m == pytest.approx(expected.pixel_width_m)
        assert geometry.height_m == pytest.approx(expected.pixel_height_m)

    def test_zero_extent_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            compute_far_field_pixel_geometry(
                PixelGeometry(width_m=75e-6, height_m=50e-6),
                ImageExtent(width_px=0, height_px=0),
                wavelength_m=1.24e-10,
                propagation_distance_m=1.0,
            )

    def test_zero_pixel_size_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            compute_far_field_pixel_geometry(
                PixelGeometry(width_m=0.0, height_m=0.0),
                ImageExtent(width_px=64, height_px=64),
                wavelength_m=1.24e-10,
                propagation_distance_m=1.0,
            )


# ---------------------------------------------------------------------------
# Direction semantics: the sign of the distance selects forward vs. backward
# ---------------------------------------------------------------------------

# Sample-plane pitch and the conjugate detector pitch it maps to, for N=128 at
# lambda=500 nm over 50 mm. The pitch passed to PropagatorParameters always names the
# *upstream* plane, so both directions use _UPSTREAM_PITCH_M.
_ROUNDTRIP_NUM_PX = 128
_ROUNDTRIP_WAVELENGTH_M = 500e-9
_ROUNDTRIP_DISTANCE_M = 0.05
_UPSTREAM_PITCH_M = 1e-5
_CONJUGATE_PITCH_M = (
    _ROUNDTRIP_WAVELENGTH_M * _ROUNDTRIP_DISTANCE_M / (_ROUNDTRIP_NUM_PX * _UPSTREAM_PITCH_M)
)


def _roundtrip_params(
    distance_m: float, pitch_m: float = _UPSTREAM_PITCH_M
) -> PropagatorParameters:
    return _make_params(
        distance_m,
        width_px=_ROUNDTRIP_NUM_PX,
        height_px=_ROUNDTRIP_NUM_PX,
        wavelength_m=_ROUNDTRIP_WAVELENGTH_M,
        pixel_width_m=pitch_m,
        pixel_height_m=pitch_m,
    )


def _apodized_noise(num_px: int = _ROUNDTRIP_NUM_PX, *, seed: int = 0) -> numpy.ndarray:
    """Complex noise under a Gaussian envelope: broadband, but negligible at the edges
    so neither plane's grid wraps."""
    rng = numpy.random.default_rng(seed)
    field = rng.normal(size=(num_px, num_px)) + 1j * rng.normal(size=(num_px, num_px))
    coord = numpy.arange(num_px) - num_px / 2
    return field * numpy.exp(-(coord[:, None] ** 2 + coord[None, :] ** 2) / 400)


@pytest.mark.parametrize('propagator_type', [FresnelTransformPropagator, FraunhoferPropagator])
class TestSingleFftDirectionSemantics:
    """A negative propagation distance must invert the forward operator built from the
    same parameters. This is what makes the propagation-distance sign meaningful, and
    it is invisible to amplitude-only comparisons: the backward branch's final multiply
    (_B) has unit modulus, so a sign error there never shows up in |output|."""

    def test_negative_distance_inverts_forward(self, propagator_type: type) -> None:
        source = _apodized_noise()
        forward = propagator_type(_roundtrip_params(+_ROUNDTRIP_DISTANCE_M))
        backward = propagator_type(_roundtrip_params(-_ROUNDTRIP_DISTANCE_M))
        recovered = backward.propagate(forward.propagate(source))
        numpy.testing.assert_allclose(recovered, source, rtol=0, atol=1e-12)

    def test_recovers_phase_not_just_amplitude(self, propagator_type: type) -> None:
        """Guards specifically against a sign error in the Fresnel number: with |Fr| in
        the phase terms the amplitude still round-trips, but the phase does not."""
        source = _apodized_noise()
        forward = propagator_type(_roundtrip_params(+_ROUNDTRIP_DISTANCE_M))
        backward = propagator_type(_roundtrip_params(-_ROUNDTRIP_DISTANCE_M))
        recovered = backward.propagate(forward.propagate(source))
        mask = numpy.abs(source) > 1e-6 * numpy.abs(source).max()
        numpy.testing.assert_allclose(
            numpy.angle(recovered[mask]), numpy.angle(source[mask]), atol=1e-9
        )

    def test_conjugate_pitch_does_not_invert(self, propagator_type: type) -> None:
        """Pins the upstream-plane convention: parameterizing the backward pass with the
        *other* plane's pitch is a real error, not a harmless relabeling."""
        source = _apodized_noise()
        forward = propagator_type(_roundtrip_params(+_ROUNDTRIP_DISTANCE_M))
        backward = propagator_type(
            _roundtrip_params(-_ROUNDTRIP_DISTANCE_M, pitch_m=_CONJUGATE_PITCH_M)
        )
        recovered = backward.propagate(forward.propagate(source))
        relative_error = numpy.linalg.norm(recovered - source) / numpy.linalg.norm(source)
        assert relative_error > 0.5

    def test_forward_branch_is_unaffected_by_the_sign_convention(
        self, propagator_type: type
    ) -> None:
        """Only the backward branch changes under the signed Fresnel number.

        For z > 0 the signed value is already positive, so the ``numpy.absolute`` the
        amplitude prefactor applies is a no-op and every forward operator is unchanged
        bit-for-bit. Asserting exact equality (not approx) is the whole point: it is
        what rules out a perturbation of the forward path.
        """
        params = _roundtrip_params(+_ROUNDTRIP_DISTANCE_M)
        fresnel_number = params.pixel_fresnel_number_x
        assert fresnel_number > 0.0
        assert numpy.absolute(fresnel_number) == fresnel_number

    def test_forward_and_backward_prefactors_are_reciprocal(self, propagator_type: type) -> None:
        """The algebraic heart of the fix, stated directly.

        Negating the distance conjugates C1 and C2 (both pure phases in the signed
        Fresnel number) and leaves the |Fr| amplitude prefactor C0 alone, so the forward
        and backward A factors multiply to exactly one. Under an absolute-valued Fresnel
        number C2 fails to conjugate and the product is not unity.
        """
        forward = propagator_type(_roundtrip_params(+_ROUNDTRIP_DISTANCE_M))
        backward = propagator_type(_roundtrip_params(-_ROUNDTRIP_DISTANCE_M))
        numpy.testing.assert_allclose(forward._A * backward._A, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Anisotropic single-FFT propagation
# ---------------------------------------------------------------------------


# ar = 2. Both axes reach the far-field/angular-spectrum crossover together when
# N*pw^2 == M*ph^2 == lambda*z, which is what lets the two propagators be compared
# on a shared grid below.
_ANISO_WAVELENGTH_M = 500e-9
_ANISO_PIXEL_WIDTH_M = 50e-6
_ANISO_PIXEL_HEIGHT_M = 25e-6
_ANISO_WIDTH_PX = 32
_ANISO_HEIGHT_PX = 128
_ANISO_CROSSOVER_M = _ANISO_WIDTH_PX * _ANISO_PIXEL_WIDTH_M**2 / _ANISO_WAVELENGTH_M


def _aniso_params(distance_m: float) -> PropagatorParameters:
    return PropagatorParameters(
        wavelength_m=_ANISO_WAVELENGTH_M,
        width_px=_ANISO_WIDTH_PX,
        height_px=_ANISO_HEIGHT_PX,
        pixel_width_m=_ANISO_PIXEL_WIDTH_M,
        pixel_height_m=_ANISO_PIXEL_HEIGHT_M,
        propagation_distance_m=distance_m,
    )


def _aniso_gaussian(params: PropagatorParameters, sigma_px: float = 3.0) -> numpy.ndarray:
    YY, XX = params.get_spatial_coordinates()  # noqa: N806
    return numpy.exp(-(numpy.square(XX) + numpy.square(YY)) / (2.0 * sigma_px**2)).astype(complex)


class TestAnisotropicSingleFftPropagation:
    """The single-FFT propagators handle non-square pixels by pairing
    ``pixel_fresnel_number_x`` -- a width-only quantity -- with ``pixel_aspect_ratio``
    at every use, so that each term recovers its correct per-axis form.

    Nothing else in the suite propagates a wavefield at ``ar != 1``, and the obvious
    tests are blind to an error here: the aspect ratio cancels between the forward and
    backward branches, so a round trip still inverts exactly, and peak-normalized
    comparisons cancel the amplitude prefactor. These therefore check absolute,
    un-normalized references, one per term that carries an aspect-ratio factor.
    """

    def test_the_geometry_is_actually_anisotropic(self) -> None:
        """Guard the premise: these tests say nothing if the pixels turn out square."""
        assert _aniso_params(1.0).pixel_aspect_ratio == pytest.approx(2.0)

    def test_amplitude_prefactor_conserves_energy_across_the_planes(self) -> None:
        """Pins ``C0``, which is ``abs(Fr) / (1j * ar)`` and must equal ``pw ph / lambda z``.

        Power is invariant once each plane is weighted by its own pixel area. Dropping
        the aspect ratio scales the recovered power by ``ar**2`` -- a factor of 4 here.
        """
        params = _aniso_params(_ANISO_CROSSOVER_M)
        wavefield = _aniso_gaussian(params)
        propagated = FresnelTransformPropagator(params).propagate(wavefield)

        conjugate = compute_far_field_pixel_geometry(
            PixelGeometry(width_m=params.pixel_width_m, height_m=params.pixel_height_m),
            ImageExtent(width_px=params.width_px, height_px=params.height_px),
            wavelength_m=params.wavelength_m,
            propagation_distance_m=params.propagation_distance_m,
        )
        power_in = numpy.sum(intensity(wavefield)) * params.pixel_width_m * params.pixel_height_m
        power_out = numpy.sum(intensity(propagated)) * conjugate.width_m * conjugate.height_m

        assert power_out == pytest.approx(power_in, rel=1e-9)

    def test_output_quadratic_phase_curves_independently_per_axis(self) -> None:
        """Pins ``C2``, the output-plane quadratic phase, which is a *pure* phase and so
        invisible to any magnitude comparison.

        Fraunhofer rather than the Fresnel transform: it carries no input chirp, so for
        a real even input the transform is real and the output phase is ``C2`` alone.
        The distance is chosen to keep the phase span near a radian; at a realistic
        far-field distance it wraps some 10^5 times and no comparison survives.
        """
        distance_m = 4.0 * _ANISO_PIXEL_WIDTH_M**2 / (numpy.pi * _ANISO_WAVELENGTH_M)
        params = _aniso_params(distance_m)
        YY, XX = params.get_spatial_coordinates()  # noqa: N806
        propagated = FraunhoferPropagator(params).propagate(_aniso_gaussian(params))

        center = (params.height_px // 2, params.width_px // 2)
        relative = numpy.angle(propagated * numpy.conjugate(propagated[center]))
        expected = (
            numpy.pi
            * params.wavelength_m
            * distance_m
            * (
                numpy.square(XX / (params.width_px * params.pixel_width_m))
                + numpy.square(YY / (params.height_px * params.pixel_height_m))
            )
        )
        expected = expected - expected[center]

        # The spectrum tails dip just below zero around 1e-6 of peak, where the phase
        # flips by pi; compare only where there is signal to carry a phase.
        carries_signal = numpy.abs(propagated) > 1e-3 * numpy.abs(propagated).max()
        residual = numpy.abs(numpy.angle(numpy.exp(1j * (relative - expected))))

        assert numpy.max(residual[carries_signal]) < 1e-9

    def test_agrees_with_angular_spectrum_at_the_crossover(self) -> None:
        """Pins the whole operator, and ``_B`` in particular, against an exact reference.

        At the crossover the two propagators share a grid and agree closely, and angular
        spectrum reaches the y-axis through ``pixel_aspect_ratio`` alone -- it never
        reads the pixel Fresnel number -- so it is an independent witness. Compared as
        complex fields, so a prefactor, an output phase or an input chirp error all show.
        """
        params = _aniso_params(_ANISO_CROSSOVER_M)
        wavefield = _aniso_gaussian(params)

        exact = AngularSpectrumPropagator(params).propagate(wavefield)
        single_fft = FresnelTransformPropagator(params).propagate(wavefield)

        scale = numpy.abs(exact).max()
        assert numpy.abs(single_fft - exact).max() / scale < 1e-5


class TestComputeNearFieldPixelGeometry:
    """The geometric back-projection of detector pixels onto the object plane.

    Pure projection: no wavelength, no distance. Its defining case is the one the
    far-field relation cannot express -- a parallel beam, where the two planes share
    a grid.
    """

    def test_unity_magnification_is_the_identity(self) -> None:
        source = PixelGeometry(width_m=75e-6, height_m=50e-6)

        geometry = compute_near_field_pixel_geometry(source, magnification=1.0)

        assert geometry.width_m == pytest.approx(source.width_m)
        assert geometry.height_m == pytest.approx(source.height_m)

    def test_demagnifies_both_axes(self) -> None:
        geometry = compute_near_field_pixel_geometry(
            PixelGeometry(width_m=75e-6, height_m=50e-6), magnification=200.0
        )

        assert geometry.width_m == pytest.approx(75e-6 / 200.0)
        assert geometry.height_m == pytest.approx(50e-6 / 200.0)

    def test_maps_each_axis_independently(self) -> None:
        """An anisotropic detector pixel stays anisotropic by the same ratio."""
        source = PixelGeometry(width_m=75e-6, height_m=50e-6)

        geometry = compute_near_field_pixel_geometry(source, magnification=4.0)

        assert geometry.width_m / geometry.height_m == pytest.approx(
            source.width_m / source.height_m
        )

    def test_agrees_with_probe_geometry_from_near_field(self) -> None:
        detector = PixelGeometry(width_m=75e-6, height_m=50e-6)
        extent = ImageExtent(width_px=256, height_px=192)
        expected = ProbeGeometry.from_near_field(detector, extent, magnification=12.0)

        geometry = compute_near_field_pixel_geometry(detector, magnification=12.0)

        assert geometry.width_m == pytest.approx(expected.pixel_width_m)
        assert geometry.height_m == pytest.approx(expected.pixel_height_m)

    def test_zero_magnification_raises(self) -> None:
        """M = 0 puts the detector at the focus, where the projection is undefined."""
        with pytest.raises(ZeroDivisionError):
            compute_near_field_pixel_geometry(
                PixelGeometry(width_m=75e-6, height_m=50e-6), magnification=0.0
            )


class TestComputeFullApertureFresnelNumber:
    """The propagation-regime indicator, and the full-aperture half of the repo's two
    Fresnel quantities.
    """

    def test_matches_closed_form(self) -> None:
        number = compute_full_aperture_fresnel_number(
            PixelGeometry(width_m=75e-6, height_m=50e-6),
            ImageExtent(width_px=256, height_px=192),
            wavelength_m=1.24e-10,
            propagation_distance_m=1.0,
        )

        expected = (256 * 75e-6) * (192 * 50e-6) / (1.24e-10 * 1.0)
        assert number == pytest.approx(expected)

    def test_reports_far_field_for_a_small_aperture_at_a_long_distance(self) -> None:
        number = compute_full_aperture_fresnel_number(
            PixelGeometry(width_m=1e-8, height_m=1e-8),
            ImageExtent(width_px=16, height_px=16),
            wavelength_m=1.24e-10,
            propagation_distance_m=1.0,
        )

        assert number < 1.0

    def test_reports_near_field_for_a_large_aperture_at_a_short_distance(self) -> None:
        number = compute_full_aperture_fresnel_number(
            PixelGeometry(width_m=75e-6, height_m=75e-6),
            ImageExtent(width_px=256, height_px=256),
            wavelength_m=1.24e-10,
            propagation_distance_m=1e-3,
        )

        assert number > 1.0

    def test_differs_from_the_pixel_number_by_the_documented_factor(self) -> None:
        """Both docstrings state the conversion; pin it so the two cannot drift."""
        pixel_geometry = PixelGeometry(width_m=75e-6, height_m=50e-6)
        extent = ImageExtent(width_px=256, height_px=192)
        parameters = PropagatorParameters(
            wavelength_m=1.24e-10,
            width_px=extent.width_px,
            height_px=extent.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            propagation_distance_m=1.0,
        )

        full_aperture = compute_full_aperture_fresnel_number(
            pixel_geometry,
            extent,
            wavelength_m=parameters.wavelength_m,
            propagation_distance_m=parameters.propagation_distance_m,
        )

        factor = (
            extent.width_px * extent.height_px * (pixel_geometry.height_m / pixel_geometry.width_m)
        )
        assert full_aperture == pytest.approx(parameters.pixel_fresnel_number_x * factor)

    def test_zero_distance_raises(self) -> None:
        with pytest.raises(ZeroDivisionError):
            compute_full_aperture_fresnel_number(
                PixelGeometry(width_m=75e-6, height_m=50e-6),
                ImageExtent(width_px=256, height_px=192),
                wavelength_m=1.24e-10,
                propagation_distance_m=0.0,
            )


# ---------------------------------------------------------------------------
# choose_propagator
# ---------------------------------------------------------------------------


def _critical_distance_m(params: PropagatorParameters) -> float:
    """Distance at which the far-field pitch equals the source pitch, i.e. where the
    two sampling regimes meet: lambda z / (N dx) == dx."""
    return params.width_px * params.pixel_width_m**2 / params.wavelength_m


class TestChoosePropagator:
    @pytest.mark.parametrize('pitch_ratio', [0.25, 0.5, 1.0])
    def test_selects_angular_spectrum_at_or_below_the_crossover(self, pitch_ratio: float) -> None:
        reference = _make_params(1.0, width_px=128, height_px=128)
        distance_m = pitch_ratio * _critical_distance_m(reference)
        params = _make_params(distance_m, width_px=128, height_px=128)
        propagator, pixel_geometry = choose_propagator(params)
        assert isinstance(propagator, AngularSpectrumPropagator)
        assert pixel_geometry.width_m == pytest.approx(params.pixel_width_m)
        assert pixel_geometry.height_m == pytest.approx(params.pixel_height_m)

    @pytest.mark.parametrize('pitch_ratio', [2.0, 4.0])
    def test_selects_fresnel_transform_above_the_crossover(self, pitch_ratio: float) -> None:
        reference = _make_params(1.0, width_px=128, height_px=128)
        distance_m = pitch_ratio * _critical_distance_m(reference)
        params = _make_params(distance_m, width_px=128, height_px=128)
        propagator, pixel_geometry = choose_propagator(params)
        assert isinstance(propagator, FresnelTransformPropagator)
        assert pixel_geometry.width_m == pytest.approx(pitch_ratio * params.pixel_width_m)

    def test_methods_agree_at_the_crossover(self) -> None:
        """The threshold is meaningful only because the two methods are numerically
        interchangeable where they meet. Away from it they differ by tens of percent,
        but that is grid disagreement, not physics."""
        num_px = 128
        params = _make_params(1.0, width_px=num_px, height_px=num_px)
        params = _make_params(_critical_distance_m(params), width_px=num_px, height_px=num_px)
        wavefield = _gaussian_wavefield(params, sigma_px=8.0)
        angular = numpy.abs(AngularSpectrumPropagator(params).propagate(wavefield))
        fresnel = numpy.abs(FresnelTransformPropagator(params).propagate(wavefield))
        numpy.testing.assert_allclose(angular / angular.max(), fresnel / fresnel.max(), atol=1e-6)

    def test_zero_distance_selects_the_identity_on_the_source_grid(self) -> None:
        """z=0 needs no special case: the far-field pitch vanishes, which is at or below
        the source pitch, so angular spectrum wins and is the identity there."""
        params = _make_params(0.0, width_px=64, height_px=64)
        propagator, pixel_geometry = choose_propagator(params)
        assert isinstance(propagator, AngularSpectrumPropagator)
        assert pixel_geometry.width_m == pytest.approx(params.pixel_width_m)
        wavefield = _gaussian_wavefield(params)
        numpy.testing.assert_allclose(propagator.propagate(wavefield), wavefield, atol=1e-12)

    def test_reported_geometry_matches_from_far_field(self) -> None:
        params = _make_params(10.0, width_px=64, height_px=64)
        propagator, pixel_geometry = choose_propagator(params)
        assert isinstance(propagator, FresnelTransformPropagator)
        expected = ProbeGeometry.from_far_field(
            PixelGeometry(width_m=params.pixel_width_m, height_m=params.pixel_height_m),
            ImageExtent(width_px=params.width_px, height_px=params.height_px),
            wavelength_m=params.wavelength_m,
            distance_m=params.propagation_distance_m,
        )
        assert pixel_geometry.width_m == pytest.approx(expected.pixel_width_m)
        assert pixel_geometry.height_m == pytest.approx(expected.pixel_height_m)

    def test_negative_distance_selects_on_magnitude(self) -> None:
        forward = choose_propagator(_make_params(+10.0, width_px=64, height_px=64))
        backward = choose_propagator(_make_params(-10.0, width_px=64, height_px=64))
        assert type(forward[0]) is type(backward[0])
        assert forward[1] == backward[1]

    def test_anisotropic_geometry_takes_the_conservative_axis(self) -> None:
        """The height axis alone is undersampled here; selecting per-axis-independently
        would alias it, so the conservative (Fresnel transform) outcome must win."""
        params = _make_params(
            0.5, width_px=128, height_px=96, pixel_width_m=50e-6, pixel_height_m=20e-6
        )
        source = PixelGeometry(width_m=params.pixel_width_m, height_m=params.pixel_height_m)
        far_field = compute_far_field_pixel_geometry(
            source,
            ImageExtent(width_px=params.width_px, height_px=params.height_px),
            wavelength_m=params.wavelength_m,
            propagation_distance_m=params.propagation_distance_m,
        )
        assert far_field.width_m <= source.width_m
        assert far_field.height_m > source.height_m

        propagator, pixel_geometry = choose_propagator(params)
        assert isinstance(propagator, FresnelTransformPropagator)
        assert pixel_geometry == far_field


class TestComputeMagnification:
    """`focus_object_distance_m` is a signed beamline coordinate, not a distance:
    its sign chooses between a diverging and a converging illumination, which are
    different geometries rather than a sign flip on one answer.

    Sibling of `compute_far_field_pixel_geometry`: this is the object-plane sampling
    relation for a cone beam, that one for the parallel-beam limit.
    """

    def test_no_focusing_optic_is_unity(self) -> None:
        assert compute_magnification(1.0, 0.0) == 1.0

    def test_converging_beam(self) -> None:
        """Focus 5 mm downstream: M = (1.0 - 0.005) / 0.005 = 199."""
        assert compute_magnification(1.0, 5e-3) == pytest.approx(199.0)

    def test_diverging_beam(self) -> None:
        """Focus 5 mm upstream: M = (1.0 + 0.005) / 0.005 = 201."""
        assert compute_magnification(1.0, -5e-3) == pytest.approx(201.0)

    def test_the_two_geometries_are_not_a_sign_flip_of_each_other(self) -> None:
        assert compute_magnification(1.0, 5e-3) != pytest.approx(compute_magnification(1.0, -5e-3))

    def test_is_never_negative(self) -> None:
        for focus_m in (-1e-3, -0.5, 0.5, 2.0):
            assert compute_magnification(1.0, focus_m) >= 0.0

    def test_detector_at_the_focus_is_zero(self) -> None:
        """Degenerate rather than an error; callers degrade on it."""
        assert compute_magnification(1.0, 1.0) == 0.0
