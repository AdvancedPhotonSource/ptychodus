"""Unit tests for wavefield propagators in ptychodus.api.propagator."""

import numpy
import numpy.testing
import pytest

from ptychodus.api.propagator import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelTransferFunctionPropagator,
    FresnelTransformPropagator,
    PropagatorParameters,
    intensity,
)


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

    def test_fresnel_number(self) -> None:
        # dx=100, z=0.1/500e-9=2e5  →  Fr = 100²/2e5 = 0.05
        params = _make_params(0.1, wavelength_m=500e-9, pixel_width_m=50e-6)
        assert params.fresnel_number == pytest.approx(0.05)

    def test_fresnel_number_uses_absolute_distance(self) -> None:
        params_pos = _make_params(+0.1)
        params_neg = _make_params(-0.1)
        assert params_pos.fresnel_number == pytest.approx(params_neg.fresnel_number)

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
        numpy.testing.assert_array_equal(prop._transfer_function[evanescent], 0.0)

    def test_transfer_function_unit_magnitude_for_propagating_modes(self) -> None:
        """TF has |TF|=1 for all propagating spatial frequencies."""
        params = _make_params(0.01)
        prop = AngularSpectrumPropagator(params)
        FY, FX = params.get_frequency_coordinates()  # noqa: N806
        ar = params.pixel_aspect_ratio
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        propagating = F2 / numpy.square(params.dx) < 1
        tf_mag = numpy.abs(prop._transfer_function[propagating])
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

    def test_is_forward_zero_distance(self) -> None:
        assert FresnelTransformPropagator(_make_params(0.0))._is_forward is True

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

    def test_is_forward_zero_distance(self) -> None:
        assert FraunhoferPropagator(_make_params(0.0))._is_forward is True

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
