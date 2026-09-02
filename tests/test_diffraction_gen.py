"""Unit tests for generate_diffraction_data in ptychodus.api.simulate.diffraction.

Behaviors verified:
  - Output shapes, dtypes, and basic invariants (non-negativity, no bad pixels)
  - Detector pixel geometry formula (λz / probe_width)
  - Flat object at integer position reproduces |Fraunhofer(probe)|²
  - Subpixel Fourier shift alters the pattern for a non-symmetric probe
  - Incoherent modes accumulate as an intensity sum
  - Poisson noise: applied only when rng is provided, reproducible with fixed seed
  - Multislice: zero inter-layer spacing is the identity; non-zero spacing changes the pattern
"""

import numpy
import numpy.testing
import pytest

from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import FraunhoferPropagator, PropagatorParameters


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

_PROBE_ENERGY_EV = 10_000.0  # 10 keV X-rays
_DETECTOR_DISTANCE_M = 1.0  # 1 m
_PIXEL_SIZE_M = 75e-9  # 75 nm object/probe pixel
_PROBE_PX = 16  # 16×16 probe
_OBJECT_PX = 64  # 64×64 object (must exceed probe size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metadata() -> ProductMetadata:
    return ProductMetadata(
        name='test',
        comments='',
        detector_distance_m=_DETECTOR_DISTANCE_M,
        probe_energy_eV=_PROBE_ENERGY_EV,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_kg=0.0,
        tomography_angle_deg=0.0,
    )


def _pixel_geometry() -> PixelGeometry:
    return PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)


def _make_probe_seq(array: numpy.ndarray) -> ProbeSequence:
    return ProbeSequence(array=array, opr_weights=None, pixel_geometry=_pixel_geometry())


def _random_probe(rng: numpy.random.Generator, num_modes: int = 1) -> numpy.ndarray:
    shape = (num_modes, _PROBE_PX, _PROBE_PX)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(complex)


def _flat_object(num_layers: int = 1, layer_spacing_m: list[float] = []) -> Object:
    layer = numpy.ones((_OBJECT_PX, _OBJECT_PX), dtype=complex)
    array = numpy.stack([layer] * num_layers) if num_layers > 1 else layer
    return Object(
        array=array,
        pixel_geometry=_pixel_geometry(),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=layer_spacing_m,
    )


def _make_object(array: numpy.ndarray, layer_spacing_m: list[float] = []) -> Object:
    return Object(
        array=array,
        pixel_geometry=_pixel_geometry(),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=layer_spacing_m,
    )


def _make_product(
    positions: list[ProbePosition],
    probe_array: numpy.ndarray,
    object_: Object | None = None,
) -> Product:
    if object_ is None:
        object_ = _flat_object()
    return Product(
        metadata=_metadata(),
        probe_positions=ProbePositionSequence(positions),
        probes=_make_probe_seq(probe_array),
        object_=object_,
        losses=[],
    )


def _center_pos(index: int = 0) -> ProbePosition:
    """Probe at the object center, which maps to an exact integer object pixel (dx=dy=0)."""
    return ProbePosition(index=index, x_m=0.0, y_m=0.0)


def _fraunhofer_intensity(probe_mode: numpy.ndarray, product: Product) -> numpy.ndarray:
    """Return |FraunhoferPropagator.propagate(probe_mode)|² using the same parameters."""
    metadata = product.metadata
    probe_geometry = product.probes.get_geometry()
    params = PropagatorParameters(
        wavelength_m=metadata.probe_wavelength_m,
        width_px=probe_geometry.width_px,
        height_px=probe_geometry.height_px,
        pixel_width_m=probe_geometry.pixel_width_m,
        pixel_height_m=probe_geometry.pixel_height_m,
        propagation_distance_m=metadata.detector_distance_m,
    )
    return numpy.square(numpy.abs(FraunhoferPropagator(params).propagate(probe_mode)))


# ---------------------------------------------------------------------------
# Output shapes and basic invariants
# ---------------------------------------------------------------------------


class TestOutputProperties:
    def test_patterns_shape(self) -> None:
        probe = _random_probe(numpy.random.default_rng(0))
        result = generate_diffraction_data(_make_product([_center_pos()], probe))
        assert result.get_patterns_shape() == (1, _PROBE_PX, _PROBE_PX)

    def test_patterns_nonnegative(self) -> None:
        probe = _random_probe(numpy.random.default_rng(1))
        result = generate_diffraction_data(_make_product([_center_pos()], probe))
        assert numpy.all(result.get_pattern(0) >= 0.0)

    def test_bad_pixels_all_false(self) -> None:
        probe = _random_probe(numpy.random.default_rng(2))
        result = generate_diffraction_data(_make_product([_center_pos()], probe))
        assert not numpy.any(result.get_bad_pixels())

    def test_pixel_geometry_formula(self) -> None:
        """Detector pixel size equals λz / probe_extent (Fraunhofer reciprocal relation)."""
        probe = _random_probe(numpy.random.default_rng(3))
        product = _make_product([_center_pos()], probe)
        result = generate_diffraction_data(product)

        metadata = product.metadata
        probe_geometry = product.probes.get_geometry()
        lambda_z = metadata.probe_wavelength_m * metadata.detector_distance_m
        pg = result.get_pixel_geometry()

        assert pg.width_m == pytest.approx(lambda_z / probe_geometry.width_m)
        assert pg.height_m == pytest.approx(lambda_z / probe_geometry.height_m)


# ---------------------------------------------------------------------------
# Pattern values
# ---------------------------------------------------------------------------


class TestPatternValues:
    def test_flat_object_matches_fraunhofer_of_probe(self) -> None:
        """For an all-ones object at the integer center, exit wave = probe, so the
        pattern must equal |Fraunhofer(probe)|²."""
        rng = numpy.random.default_rng(10)
        probe = _random_probe(rng)  # shape (1, H, W)
        product = _make_product([_center_pos()], probe, _flat_object())

        result = generate_diffraction_data(product)

        # probe[0] is the single incoherent mode; object patch is all ones → no effect
        expected = _fraunhofer_intensity(probe[0], product)
        numpy.testing.assert_allclose(result.get_pattern(0), expected, rtol=1e-12)

    def test_subpixel_shift_changes_pattern(self) -> None:
        """A 0.5-pixel Fourier shift of the probe changes the pattern when the object is non-flat.

        For a flat object, shifting the probe only adds a phase ramp to its FT, leaving
        |FFT(shifted_probe)|² unchanged.  With a non-uniform object the exit wave
        (shifted_probe × object_patch) does differ, producing a distinct far-field pattern.
        """
        rng = numpy.random.default_rng(11)
        probe = _random_probe(rng)

        rng_obj = numpy.random.default_rng(111)
        obj_layer = (
            rng_obj.standard_normal((_OBJECT_PX, _OBJECT_PX))
            + 1j * rng_obj.standard_normal((_OBJECT_PX, _OBJECT_PX))
        ).astype(complex)
        object_ = _make_object(obj_layer)

        pos_int = ProbePosition(index=0, x_m=0.0, y_m=0.0)
        pos_half = ProbePosition(index=0, x_m=0.5 * _PIXEL_SIZE_M, y_m=0.0)

        result_int = generate_diffraction_data(_make_product([pos_int], probe, object_))
        result_half = generate_diffraction_data(_make_product([pos_half], probe, object_))

        assert not numpy.allclose(result_int.get_pattern(0), result_half.get_pattern(0))

    def test_incoherent_modes_accumulate_as_intensity_sum(self) -> None:
        """Pattern from two incoherent modes equals the sum of each mode's pattern."""
        rng = numpy.random.default_rng(12)
        probe_two = _random_probe(rng, num_modes=2)  # shape (2, H, W)
        object_ = _flat_object()
        pos = _center_pos()

        result_both = generate_diffraction_data(_make_product([pos], probe_two, object_))
        result_m0 = generate_diffraction_data(_make_product([pos], probe_two[0:1], object_))
        result_m1 = generate_diffraction_data(_make_product([pos], probe_two[1:2], object_))

        expected = result_m0.get_pattern(0) + result_m1.get_pattern(0)
        numpy.testing.assert_allclose(result_both.get_pattern(0), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# Poisson noise
# ---------------------------------------------------------------------------


class TestPoissonNoise:
    def test_no_rng_gives_deterministic_result(self) -> None:
        """Without an rng the function is deterministic: two calls return identical patterns."""
        probe = _random_probe(numpy.random.default_rng(20))
        product = _make_product([_center_pos()], probe)
        result_a = generate_diffraction_data(product)
        result_b = generate_diffraction_data(product)
        numpy.testing.assert_array_equal(result_a.get_pattern(0), result_b.get_pattern(0))

    def test_rng_produces_nonnegative_integer_valued_floats(self) -> None:
        """Poisson samples are non-negative and integer-valued (stored as float)."""
        rng = numpy.random.default_rng(21)
        probe = _random_probe(rng) * 1e4  # amplify so counts are meaningful
        result = generate_diffraction_data(
            _make_product([_center_pos()], probe),
            rng=numpy.random.default_rng(99),
        )
        pattern = result.get_pattern(0)
        assert numpy.all(pattern >= 0.0)
        numpy.testing.assert_array_equal(pattern, numpy.floor(pattern))

    def test_noise_reproducible_with_same_seed(self) -> None:
        """Identical rng seeds yield identical noisy patterns."""
        probe = _random_probe(numpy.random.default_rng(22)) * 1e4
        product = _make_product([_center_pos()], probe)
        result_a = generate_diffraction_data(product, rng=numpy.random.default_rng(42))
        result_b = generate_diffraction_data(product, rng=numpy.random.default_rng(42))
        numpy.testing.assert_array_equal(result_a.get_pattern(0), result_b.get_pattern(0))

    def test_noise_differs_from_noiseless(self) -> None:
        """Applying Poisson noise changes the pattern values."""
        probe = _random_probe(numpy.random.default_rng(23)) * 1e4
        product = _make_product([_center_pos()], probe)
        result_clean = generate_diffraction_data(product)
        result_noisy = generate_diffraction_data(product, rng=numpy.random.default_rng(0))
        assert not numpy.array_equal(result_clean.get_pattern(0), result_noisy.get_pattern(0))


# ---------------------------------------------------------------------------
# Multislice propagation
# ---------------------------------------------------------------------------


class TestMultislice:
    def test_zero_spacing_two_layers_matches_single_layer(self) -> None:
        """Two all-ones layers with zero inter-layer spacing is equivalent to one layer.

        At spacing=0 the AngularSpectrumPropagator transfer function is exp(0)=1,
        so propagation is the identity and the multislice reduces to the single-slice result.
        """
        rng = numpy.random.default_rng(30)
        probe = _random_probe(rng)
        pos = _center_pos()

        result_one = generate_diffraction_data(_make_product([pos], probe, _flat_object(1)))
        result_two = generate_diffraction_data(_make_product([pos], probe, _flat_object(2, [0.0])))

        numpy.testing.assert_allclose(
            result_one.get_pattern(0), result_two.get_pattern(0), rtol=1e-12
        )

    def test_nonzero_spacing_changes_pattern(self) -> None:
        """Non-zero inter-layer spacing modifies the wavefield and therefore the pattern."""
        rng_probe = numpy.random.default_rng(31)
        rng_obj = numpy.random.default_rng(32)
        probe = _random_probe(rng_probe)
        pos = _center_pos()

        layer = (
            rng_obj.standard_normal((_OBJECT_PX, _OBJECT_PX))
            + 1j * rng_obj.standard_normal((_OBJECT_PX, _OBJECT_PX))
        ).astype(complex)
        two_layer = numpy.stack([layer, layer])

        object_zero = _make_object(two_layer.copy(), layer_spacing_m=[0.0])
        object_nonzero = _make_object(two_layer.copy(), layer_spacing_m=[1e-3])

        result_zero = generate_diffraction_data(_make_product([pos], probe, object_zero))
        result_nonzero = generate_diffraction_data(_make_product([pos], probe, object_nonzero))

        assert not numpy.allclose(result_zero.get_pattern(0), result_nonzero.get_pattern(0))


# ---------------------------------------------------------------------------
# Multi-position iteration (regression: zip(positions, probes) silently
# truncated to a single position when OPR weights were absent)
# ---------------------------------------------------------------------------


class TestMultiPositionWithoutOpr:
    def test_all_positions_populated_without_opr_weights(self) -> None:
        """With multiple scan positions and no OPR weights, every output pattern
        must be populated — not just the first."""
        rng = numpy.random.default_rng(40)
        probe = _random_probe(rng)
        positions = [
            ProbePosition(index=0, x_m=-8 * _PIXEL_SIZE_M, y_m=0.0),
            ProbePosition(index=1, x_m=0.0, y_m=0.0),
            ProbePosition(index=2, x_m=+8 * _PIXEL_SIZE_M, y_m=0.0),
        ]
        # _make_probe_seq constructs a ProbeSequence with opr_weights=None.
        product = _make_product(positions, probe, _flat_object())

        result = generate_diffraction_data(product)
        assert result.get_patterns_shape() == (3, _PROBE_PX, _PROBE_PX)
        # The bug symptom was patterns[1:] all zero. Assert every position has signal.
        for i in range(3):
            assert numpy.sum(result.get_pattern(i)) > 0.0
