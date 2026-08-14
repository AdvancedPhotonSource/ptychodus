"""Unit tests for ptychodus.api.reconstruct.ReconstructionAmbiguities."""

from __future__ import annotations

from pathlib import Path

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.diffraction import AssembledDiffractionData
from ptychodus.api.metrics import estimate_reconstruction_ambiguities
from ptychodus.api.reconstruct import (
    NullReconstructor,
    PositionIndexFilter,
    ReconstructionAmbiguities,
    prepare_reconstruct_input,
)


PIXEL_M = 1.0e-9  # 1 nm per pixel for both probe and object
OBJ_HEIGHT_PX = 32
OBJ_WIDTH_PX = 40
PROBE_HEIGHT_PX = 8
PROBE_WIDTH_PX = 8


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


def _make_object(num_layers: int = 1, *, seed: int = 0) -> Object:
    rng = numpy.random.default_rng(seed)
    real = rng.standard_normal((num_layers, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
    imag = rng.standard_normal((num_layers, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
    array = (real + 1j * imag).astype(numpy.complex128)
    return Object(
        array=array,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
        layer_spacing_m=[1.0e-6] * (num_layers - 1),
    )


def _make_probes(
    *, num_coherent_modes: int = 1, num_incoherent_modes: int = 1, seed: int = 1
) -> ProbeSequence:
    rng = numpy.random.default_rng(seed)
    shape = (num_coherent_modes, num_incoherent_modes, PROBE_HEIGHT_PX, PROBE_WIDTH_PX)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    array = (real + 1j * imag).astype(numpy.complex128)
    return ProbeSequence(
        array=array,
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
    )


def _make_positions() -> ProbePositionSequence:
    # A handful of positions spread around the object center.
    points = [
        ProbePosition(index=i, coordinate_x_m=x, coordinate_y_m=y)
        for i, (x, y) in enumerate(
            [(0.0, 0.0), (3 * PIXEL_M, -2 * PIXEL_M), (-4 * PIXEL_M, 1 * PIXEL_M)]
        )
    ]
    return ProbePositionSequence(points)


def _make_product(num_layers: int = 1) -> Product:
    return Product(
        metadata=_metadata(),
        probe_positions=_make_positions(),
        probes=_make_probes(),
        object_=_make_object(num_layers=num_layers),
        losses=[],
    )


def _exit_waves(product: Product) -> list[numpy.ndarray]:
    """probe(r_local) * object(r_local + r_pos) at every scan position.

    The probe coordinate frame and the object coordinate frame both use the
    geometric center of their respective arrays as the origin (matching the
    convention in ``ReconstructionAmbiguities.standardize_product``), so the
    window in object pixels is centered on the pixel nearest the position.
    """
    obj_array = product.object_.get_array()
    obj_total = numpy.prod(obj_array, axis=0)  # collapse layers
    obj_geom = product.object_.get_geometry()
    obj_center = obj_geom.get_center()
    probe_array = product.probes.get_array()
    # Use mode 0/0; the standardization multiplies every mode by the same
    # complex factor, so comparing one mode is sufficient and avoids OPR mixing.
    probe = probe_array[0, 0]

    exit_waves: list[numpy.ndarray] = []
    obj_center_x_px = (obj_geom.width_px - 1) / 2
    obj_center_y_px = (obj_geom.height_px - 1) / 2
    half_w = PROBE_WIDTH_PX // 2
    half_h = PROBE_HEIGHT_PX // 2

    for position in product.probe_positions:
        cx_px = int(
            round(obj_center_x_px + (position.coordinate_x_m - obj_center.coordinate_x_m) / PIXEL_M)
        )
        cy_px = int(
            round(obj_center_y_px + (position.coordinate_y_m - obj_center.coordinate_y_m) / PIXEL_M)
        )
        window = obj_total[
            cy_px - half_h : cy_px - half_h + PROBE_HEIGHT_PX,
            cx_px - half_w : cx_px - half_w + PROBE_WIDTH_PX,
        ]
        exit_waves.append(probe * window)

    return exit_waves


def _exit_wave_magnitudes(product: Product) -> list[numpy.ndarray]:
    return [numpy.abs(ew) for ew in _exit_waves(product)]


def _diffraction_intensities(product: Product) -> list[numpy.ndarray]:
    return [numpy.abs(numpy.fft.fft2(ew)) ** 2 for ew in _exit_waves(product)]


class TestReconstructionAmbiguitiesConstruction:
    def test_identity_factory_round_trips_the_four_fields(self) -> None:
        ambiguities = ReconstructionAmbiguities.create_identity()
        assert ambiguities.object_scale_factor == 1.0
        assert ambiguities.phase_offset_rad == 0.0
        assert ambiguities.phase_ramp_x_rad_per_m == 0.0
        assert ambiguities.phase_ramp_y_rad_per_m == 0.0

    def test_zero_scale_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='object_scale_factor'):
            ReconstructionAmbiguities(0.0, 0.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'object_scale_factor': float('nan')},
            {'object_scale_factor': float('inf')},
            {'phase_offset_rad': float('nan')},
            {'phase_offset_rad': float('inf')},
            {'phase_ramp_x_rad_per_m': float('nan')},
            {'phase_ramp_y_rad_per_m': float('inf')},
        ],
    )
    def test_nonfinite_fields_are_rejected(self, kwargs: dict[str, float]) -> None:
        base = dict(
            object_scale_factor=1.0,
            phase_offset_rad=0.0,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )
        base.update(kwargs)
        with pytest.raises(ValueError, match='must be finite'):
            ReconstructionAmbiguities(**base)


class TestStandardizeProductIdentity:
    def test_identity_passes_object_through_unchanged(self) -> None:
        product = _make_product()
        result = ReconstructionAmbiguities.create_identity().standardize_product(product)
        numpy.testing.assert_array_equal(
            result.object_.get_array(),
            product.object_.get_array(),
        )

    def test_identity_passes_probes_through_unchanged(self) -> None:
        product = _make_product()
        result = ReconstructionAmbiguities.create_identity().standardize_product(product)
        numpy.testing.assert_array_equal(
            result.probes.get_array(),
            product.probes.get_array(),
        )

    def test_identity_passes_positions_metadata_and_losses_through(self) -> None:
        product = _make_product()
        result = ReconstructionAmbiguities.create_identity().standardize_product(product)
        assert result.metadata is product.metadata
        assert result.probe_positions is product.probe_positions
        assert result.losses is product.losses


class TestStandardizeProductRoundTrip:
    """Apply an ambiguity to a clean product, then standardize and recover it."""

    @staticmethod
    def _apply_ambiguity_to_object(obj: Object, ambiguities: ReconstructionAmbiguities) -> Object:
        coords = obj.get_geometry().get_transverse_coordinates()
        ramp = (
            ambiguities.phase_ramp_x_rad_per_m * coords.position_x_m
            + ambiguities.phase_ramp_y_rad_per_m * coords.position_y_m
        )
        factor = ambiguities.object_scale_factor * numpy.exp(
            1j * (ambiguities.phase_offset_rad + ramp)
        )
        new_array = obj.get_array().copy()
        new_array[0] = (new_array[0] * factor).astype(new_array.dtype)
        return Object(
            array=new_array,
            pixel_geometry=obj.get_pixel_geometry().copy(),
            center=obj.get_center().copy(),
            layer_spacing_m=list(obj.layer_spacing_m),
        )

    def test_round_trip_recovers_object_layer_0(self) -> None:
        clean = _make_product()
        ambiguities = ReconstructionAmbiguities(
            object_scale_factor=2.5,
            phase_offset_rad=0.3,
            phase_ramp_x_rad_per_m=1.0e8,
            phase_ramp_y_rad_per_m=-2.0e8,
        )
        perturbed_object = self._apply_ambiguity_to_object(clean.object_, ambiguities)
        perturbed = Product(
            metadata=clean.metadata,
            probe_positions=clean.probe_positions,
            probes=clean.probes,
            object_=perturbed_object,
            losses=clean.losses,
        )

        recovered = ambiguities.standardize_product(perturbed)

        numpy.testing.assert_allclose(
            recovered.object_.get_array(),
            clean.object_.get_array(),
            rtol=1.0e-10,
            atol=1.0e-12,
        )


class TestStandardizeProductMultiLayer:
    """Regression: the correction must scale the *total* transmission, not each layer."""

    def test_total_transmission_scales_by_inverse_scale_factor(self) -> None:
        product = _make_product(num_layers=3)
        ambiguities = ReconstructionAmbiguities(
            object_scale_factor=4.0,
            phase_offset_rad=0.0,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )
        result = ambiguities.standardize_product(product)

        original_total = numpy.prod(product.object_.get_array(), axis=0)
        standardized_total = numpy.prod(result.object_.get_array(), axis=0)

        numpy.testing.assert_allclose(
            standardized_total,
            original_total / ambiguities.object_scale_factor,
            rtol=1.0e-12,
        )

    def test_layers_above_zero_pass_through_unchanged(self) -> None:
        product = _make_product(num_layers=3)
        ambiguities = ReconstructionAmbiguities(
            object_scale_factor=4.0,
            phase_offset_rad=1.7,
            phase_ramp_x_rad_per_m=5.0e7,
            phase_ramp_y_rad_per_m=-3.0e7,
        )
        result = ambiguities.standardize_product(product)

        numpy.testing.assert_array_equal(
            result.object_.get_array()[1:],
            product.object_.get_array()[1:],
        )


class TestStandardizeProductExitWavePreservation:
    """The contract: diffraction-pattern intensities are preserved at every scan position."""

    def test_exit_wave_magnitudes_match_with_non_zero_ramp(self) -> None:
        product = _make_product()
        ambiguities = ReconstructionAmbiguities(
            object_scale_factor=1.5,
            phase_offset_rad=0.7,
            phase_ramp_x_rad_per_m=3.0e8,
            phase_ramp_y_rad_per_m=-1.0e8,
        )

        before = _exit_wave_magnitudes(product)
        after = _exit_wave_magnitudes(ambiguities.standardize_product(product))

        assert len(before) == len(after)
        for ew_before, ew_after in zip(before, after):
            numpy.testing.assert_allclose(ew_after, ew_before, rtol=1.0e-10, atol=1.0e-12)

    def test_diffraction_intensities_match_with_non_zero_ramp(self) -> None:
        # Pixel-wise |probe * window| is insensitive to phase ramps, so it
        # cannot distinguish the correct sign on the probe's ramp correction.
        # |FFT(exit_wave)|^2 is the actual observable and is shifted by an
        # incorrect ramp sign — this is what the docstring contract requires.
        product = _make_product()
        ambiguities = ReconstructionAmbiguities(
            object_scale_factor=1.5,
            phase_offset_rad=0.7,
            phase_ramp_x_rad_per_m=3.0e8,
            phase_ramp_y_rad_per_m=-1.0e8,
        )

        before = _diffraction_intensities(product)
        after = _diffraction_intensities(ambiguities.standardize_product(product))

        assert len(before) == len(after)
        for di_before, di_after in zip(before, after):
            numpy.testing.assert_allclose(di_after, di_before, rtol=1.0e-10, atol=1.0e-12)


class TestStandardizeProductOPRPassthrough:
    """OPR weights, when present, must pass through unchanged (copied, not aliased)."""

    def test_opr_weights_preserved(self) -> None:
        rng = numpy.random.default_rng(42)
        array = (
            rng.standard_normal((2, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX))
            + 1j * rng.standard_normal((2, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX))
        ).astype(numpy.complex128)
        weights = rng.standard_normal((5, 2)).astype(numpy.float64)
        probes = ProbeSequence(
            array=array,
            opr_weights=weights,
            pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        )
        product = Product(
            metadata=_metadata(),
            probe_positions=_make_positions(),
            probes=probes,
            object_=_make_object(),
            losses=[],
        )
        ambiguities = ReconstructionAmbiguities(
            object_scale_factor=0.5,
            phase_offset_rad=1.1,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )

        result = ambiguities.standardize_product(product)
        result_weights = result.probes.get_opr_weights()

        numpy.testing.assert_array_equal(result_weights, weights)
        assert result_weights is not weights  # copied, not aliased


def _make_real_positive_object(seed: int = 7) -> Object:
    """Object whose layer 0 has zero phase everywhere — exact-recovery test fixture."""
    rng = numpy.random.default_rng(seed)
    amp = rng.uniform(0.5, 1.5, (OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
    array = amp.astype(numpy.complex128)[numpy.newaxis, ...]
    return Object(
        array=array,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
        layer_spacing_m=[],
    )


def _replace_object(product: Product, new_object: Object) -> Product:
    return Product(
        metadata=product.metadata,
        probe_positions=product.probe_positions,
        probes=product.probes,
        object_=new_object,
        losses=product.losses,
    )


class TestEstimateSingleProduct:
    """Single-product estimator: recover (phi, k_x, k_y); scale fixed at 1.0."""

    def test_scale_factor_is_always_one(self) -> None:
        product = _make_product()
        estimate = estimate_reconstruction_ambiguities(product)
        assert estimate.object_scale_factor == 1.0

    def test_real_positive_object_recovers_phi_and_ramp_exactly(self) -> None:
        # A real-positive object has zero clean phase, so the only signal driving
        # the estimator is the applied ambiguity. Recovery is exact up to fp noise.
        clean = _replace_object(_make_product(), _make_real_positive_object())
        applied = ReconstructionAmbiguities(
            object_scale_factor=1.0,
            phase_offset_rad=0.4,
            phase_ramp_x_rad_per_m=2.0e7,
            phase_ramp_y_rad_per_m=-1.0e7,
        )
        perturbed_obj = TestStandardizeProductRoundTrip._apply_ambiguity_to_object(
            clean.object_, applied
        )
        perturbed = _replace_object(clean, perturbed_obj)

        estimate = estimate_reconstruction_ambiguities(perturbed)

        assert estimate.object_scale_factor == 1.0
        numpy.testing.assert_allclose(
            estimate.phase_offset_rad, applied.phase_offset_rad, atol=1.0e-12
        )
        numpy.testing.assert_allclose(
            estimate.phase_ramp_x_rad_per_m, applied.phase_ramp_x_rad_per_m, atol=1.0e-3
        )
        numpy.testing.assert_allclose(
            estimate.phase_ramp_y_rad_per_m, applied.phase_ramp_y_rad_per_m, atol=1.0e-3
        )

    def test_estimate_then_standardize_is_idempotent(self) -> None:
        # On a random object the absolute estimate has 1/sqrt(N) noise, but
        # estimating then standardizing then re-estimating should land at identity.
        product = _make_product()
        estimate = estimate_reconstruction_ambiguities(product)
        standardized = estimate.standardize_product(product)
        re_estimate = estimate_reconstruction_ambiguities(standardized)

        assert re_estimate.object_scale_factor == 1.0
        assert abs(re_estimate.phase_offset_rad) < 1.0e-10
        # On a random complex array the differential-phase circular mean is
        # essentially zero after one canonicalization; allow tight tolerance
        # in per-pixel units, divided by pixel size to convert to rad/m.
        assert abs(re_estimate.phase_ramp_x_rad_per_m * PIXEL_M) < 1.0e-10
        assert abs(re_estimate.phase_ramp_y_rad_per_m * PIXEL_M) < 1.0e-10

    def test_standardize_with_estimate_removes_added_ramp(self) -> None:
        # End-to-end: apply ramp, estimate, standardize — the result equals the
        # standardized clean product (both go through the same canonicalization,
        # so any "baseline" of the clean object cancels).
        clean = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=1.0,
            phase_offset_rad=0.3,
            phase_ramp_x_rad_per_m=1.0e7,
            phase_ramp_y_rad_per_m=-5.0e6,
        )
        perturbed_obj = TestStandardizeProductRoundTrip._apply_ambiguity_to_object(
            clean.object_, applied
        )
        perturbed = _replace_object(clean, perturbed_obj)

        clean_canonical = estimate_reconstruction_ambiguities(clean).standardize_product(clean)
        perturbed_canonical = estimate_reconstruction_ambiguities(perturbed).standardize_product(
            perturbed
        )
        numpy.testing.assert_allclose(
            perturbed_canonical.object_.get_array()[0],
            clean_canonical.object_.get_array()[0],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    def test_multilayer_object_uses_only_layer_zero(self) -> None:
        product = _make_product(num_layers=3)
        single_layer_obj = Object(
            array=product.object_.get_array()[0:1].copy(),
            pixel_geometry=product.object_.get_pixel_geometry().copy(),
            center=product.object_.get_center().copy(),
            layer_spacing_m=[],
        )
        single_layer_product = _replace_object(product, single_layer_obj)

        multi_estimate = estimate_reconstruction_ambiguities(product)
        single_estimate = estimate_reconstruction_ambiguities(single_layer_product)
        assert multi_estimate == single_estimate

    def test_all_ones_weights_match_no_weights(self) -> None:
        product = _make_product()
        layer_shape = product.object_.get_array()[0].shape
        ones = numpy.ones(layer_shape, dtype=numpy.float64)

        unweighted = estimate_reconstruction_ambiguities(product)
        weighted = estimate_reconstruction_ambiguities(product, weights=ones)

        assert unweighted == weighted

    def test_region_mask_recovers_known_ambiguity(self) -> None:
        # On a real-positive object the unweighted estimate is exact; a 0/1
        # mask covering most of the array is also exact (the masked region
        # still carries the ramp signal uniformly).
        product = _replace_object(_make_product(), _make_real_positive_object())
        applied = ReconstructionAmbiguities(
            object_scale_factor=1.0,
            phase_offset_rad=0.2,
            phase_ramp_x_rad_per_m=5.0e6,
            phase_ramp_y_rad_per_m=-2.0e6,
        )
        perturbed_obj = TestStandardizeProductRoundTrip._apply_ambiguity_to_object(
            product.object_, applied
        )
        perturbed = _replace_object(product, perturbed_obj)

        layer_shape = perturbed.object_.get_array()[0].shape
        mask = numpy.ones(layer_shape, dtype=numpy.float64)
        mask[: layer_shape[0] // 4, :] = 0.0
        mask[:, : layer_shape[1] // 4] = 0.0

        estimate = estimate_reconstruction_ambiguities(perturbed, weights=mask)
        numpy.testing.assert_allclose(
            estimate.phase_offset_rad, applied.phase_offset_rad, atol=1.0e-10
        )
        numpy.testing.assert_allclose(
            estimate.phase_ramp_x_rad_per_m, applied.phase_ramp_x_rad_per_m, atol=1.0e-3
        )
        numpy.testing.assert_allclose(
            estimate.phase_ramp_y_rad_per_m, applied.phase_ramp_y_rad_per_m, atol=1.0e-3
        )

    def test_complex64_input_is_handled(self) -> None:
        product = _make_product()
        complex64_array = product.object_.get_array().astype(numpy.complex64)
        new_obj = Object(
            array=complex64_array,
            pixel_geometry=product.object_.get_pixel_geometry().copy(),
            center=product.object_.get_center().copy(),
            layer_spacing_m=list(product.object_.layer_spacing_m),
        )
        product_c64 = _replace_object(product, new_obj)

        estimate = estimate_reconstruction_ambiguities(product_c64)
        assert numpy.isfinite(estimate.phase_offset_rad)
        assert numpy.isfinite(estimate.phase_ramp_x_rad_per_m)
        assert numpy.isfinite(estimate.phase_ramp_y_rad_per_m)


class TestEstimateRelative:
    """Relative estimator: recover all four ambiguities aligning target to reference."""

    @staticmethod
    def _perturbed_product(clean: Product, applied: ReconstructionAmbiguities) -> Product:
        perturbed_obj = TestStandardizeProductRoundTrip._apply_ambiguity_to_object(
            clean.object_, applied
        )
        return _replace_object(clean, perturbed_obj)

    def test_identical_products_give_identity(self) -> None:
        product = _make_product()
        estimate = estimate_reconstruction_ambiguities(product, reference=product)
        numpy.testing.assert_allclose(estimate.object_scale_factor, 1.0, rtol=1.0e-12)
        assert abs(estimate.phase_offset_rad) < 1.0e-10
        # Tolerance is per-pixel: rad/m * pixel_m. The accumulated complex sum
        # has fp roundoff on the order of 1e-15 / pixel = 1e-15 / 1e-9 m = 1e-6
        # rad/m for the worst case; cap with a generous bound.
        assert abs(estimate.phase_ramp_x_rad_per_m * PIXEL_M) < 1.0e-12
        assert abs(estimate.phase_ramp_y_rad_per_m * PIXEL_M) < 1.0e-12

    def test_recovers_all_four_parameters_on_random_object(self) -> None:
        # For the relative estimator, S = target * conj(ref) cancels the
        # clean object's phase content, so recovery is precise on any
        # non-degenerate object.
        clean = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=2.5,
            phase_offset_rad=0.3,
            phase_ramp_x_rad_per_m=2.0e7,
            phase_ramp_y_rad_per_m=-1.0e7,
        )
        target = self._perturbed_product(clean, applied)

        estimate = estimate_reconstruction_ambiguities(target, reference=clean)

        numpy.testing.assert_allclose(
            estimate.object_scale_factor, applied.object_scale_factor, rtol=1.0e-10
        )
        numpy.testing.assert_allclose(
            estimate.phase_offset_rad, applied.phase_offset_rad, atol=1.0e-10
        )
        numpy.testing.assert_allclose(
            estimate.phase_ramp_x_rad_per_m, applied.phase_ramp_x_rad_per_m, atol=1.0e-3
        )
        numpy.testing.assert_allclose(
            estimate.phase_ramp_y_rad_per_m, applied.phase_ramp_y_rad_per_m, atol=1.0e-3
        )

    def test_standardize_target_with_estimate_recovers_reference(self) -> None:
        clean = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=2.5,
            phase_offset_rad=0.3,
            phase_ramp_x_rad_per_m=2.0e7,
            phase_ramp_y_rad_per_m=-1.0e7,
        )
        target = self._perturbed_product(clean, applied)

        estimate = estimate_reconstruction_ambiguities(target, reference=clean)
        recovered = estimate.standardize_product(target)

        numpy.testing.assert_allclose(
            recovered.object_.get_array()[0],
            clean.object_.get_array()[0],
            rtol=1.0e-9,
            atol=1.0e-11,
        )

    def test_negative_scale_is_folded_into_phase(self) -> None:
        # Applying a negative real scale to a clean object is indistinguishable
        # from positive scale + pi phase offset. The estimator must return the
        # positive-scale form.
        clean = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=-1.5,
            phase_offset_rad=0.0,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )
        target = self._perturbed_product(clean, applied)

        estimate = estimate_reconstruction_ambiguities(target, reference=clean)

        assert estimate.object_scale_factor > 0.0
        numpy.testing.assert_allclose(estimate.object_scale_factor, 1.5, rtol=1.0e-10)
        # phi should be equivalent to pi modulo 2*pi. Reduce to [-pi, pi] via
        # phi mod 2pi - pi, then compare to 0.
        wrapped = estimate.phase_offset_rad % (2.0 * numpy.pi) - numpy.pi
        assert abs(wrapped) < 1.0e-10

    def test_all_ones_weights_match_no_weights(self) -> None:
        clean = _make_product()
        applied = ReconstructionAmbiguities(
            object_scale_factor=2.0,
            phase_offset_rad=0.4,
            phase_ramp_x_rad_per_m=1.0e7,
            phase_ramp_y_rad_per_m=-2.0e7,
        )
        target = self._perturbed_product(clean, applied)
        ones = numpy.ones(clean.object_.get_array()[0].shape, dtype=numpy.float64)

        without = estimate_reconstruction_ambiguities(target, reference=clean)
        with_ones = estimate_reconstruction_ambiguities(target, reference=clean, weights=ones)

        assert without == with_ones

    def test_layer_zero_only_on_multilayer_objects(self) -> None:
        clean = _make_product(num_layers=3)
        applied = ReconstructionAmbiguities(
            object_scale_factor=1.5,
            phase_offset_rad=0.2,
            phase_ramp_x_rad_per_m=5.0e6,
            phase_ramp_y_rad_per_m=-3.0e6,
        )
        target = self._perturbed_product(clean, applied)

        def first_layer_product(p: Product) -> Product:
            new_obj = Object(
                array=p.object_.get_array()[0:1].copy(),
                pixel_geometry=p.object_.get_pixel_geometry().copy(),
                center=p.object_.get_center().copy(),
                layer_spacing_m=[],
            )
            return _replace_object(p, new_obj)

        multi_estimate = estimate_reconstruction_ambiguities(target, reference=clean)
        single_estimate = estimate_reconstruction_ambiguities(
            first_layer_product(target), reference=first_layer_product(clean)
        )
        assert multi_estimate == single_estimate

    def test_shape_mismatch_raises(self) -> None:
        product_a = _make_product()
        rng = numpy.random.default_rng(99)
        small_array = (
            rng.standard_normal((1, 16, 20)) + 1j * rng.standard_normal((1, 16, 20))
        ).astype(numpy.complex128)
        small_obj = Object(
            array=small_array,
            pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
            center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
            layer_spacing_m=[],
        )
        product_b = _replace_object(product_a, small_obj)

        with pytest.raises(ValueError, match='shape'):
            estimate_reconstruction_ambiguities(product_b, reference=product_a)

    def test_pixel_geometry_mismatch_raises(self) -> None:
        product_a = _make_product()
        rng = numpy.random.default_rng(123)
        array = (
            rng.standard_normal((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
            + 1j * rng.standard_normal((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
        ).astype(numpy.complex128)
        different_pixel_obj = Object(
            array=array,
            pixel_geometry=PixelGeometry(width_m=2.0 * PIXEL_M, height_m=PIXEL_M),
            center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
            layer_spacing_m=[],
        )
        product_b = _replace_object(product_a, different_pixel_obj)

        with pytest.raises(ValueError, match='pixel geometry'):
            estimate_reconstruction_ambiguities(product_b, reference=product_a)

    def test_zero_reference_raises(self) -> None:
        clean = _make_product()
        zero_obj = Object(
            array=numpy.zeros((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX), dtype=numpy.complex128),
            pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
            center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
            layer_spacing_m=[],
        )
        zero_product = _replace_object(clean, zero_obj)
        with pytest.raises(ValueError, match='zero'):
            estimate_reconstruction_ambiguities(clean, reference=zero_product)


class TestEstimateValidation:
    """Validation of the optional weights argument shared by both estimators."""

    def test_weights_shape_mismatch_raises(self) -> None:
        product = _make_product()
        bad_weights = numpy.ones((1, 1), dtype=numpy.float64)
        with pytest.raises(ValueError, match='shape'):
            estimate_reconstruction_ambiguities(product, weights=bad_weights)

    def test_negative_weights_raises(self) -> None:
        product = _make_product()
        bad_weights = -numpy.ones(product.object_.get_array()[0].shape, dtype=numpy.float64)
        with pytest.raises(ValueError, match='non-negative'):
            estimate_reconstruction_ambiguities(product, weights=bad_weights)

    def test_non_finite_weights_raises(self) -> None:
        product = _make_product()
        bad_weights = numpy.ones(product.object_.get_array()[0].shape, dtype=numpy.float64)
        bad_weights[0, 0] = numpy.nan
        with pytest.raises(ValueError, match='finite'):
            estimate_reconstruction_ambiguities(product, weights=bad_weights)

    def test_all_zero_weights_raises_for_single_product(self) -> None:
        product = _make_product()
        zeros = numpy.zeros(product.object_.get_array()[0].shape, dtype=numpy.float64)
        with pytest.raises(ValueError, match='zero'):
            estimate_reconstruction_ambiguities(product, weights=zeros)

    def test_all_zero_weights_raises_for_relative(self) -> None:
        product = _make_product()
        zeros = numpy.zeros(product.object_.get_array()[0].shape, dtype=numpy.float64)
        with pytest.raises(ValueError, match='zero'):
            estimate_reconstruction_ambiguities(product, reference=product, weights=zeros)


def _make_assembled_data(indexes: list[int], pattern_hw: int = 4) -> AssembledDiffractionData:
    """Construct an AssembledDiffractionData whose pattern rows are tagged with their index.

    ``AssembledDiffractionData.get_indexes()`` filters out the ``< 0`` sentinel, so
    passing all-negative indexes yields a dataset that appears empty to the matcher.
    Each pattern row is filled with its index value so tests can assert which row
    survived the merge by reading any pixel.
    """
    indexes_arr = numpy.asarray(indexes, dtype=numpy.intp)
    n = indexes_arr.size
    patterns = numpy.broadcast_to(indexes_arr.reshape(n, 1, 1), (n, pattern_hw, pattern_hw)).astype(
        numpy.intp
    )
    return AssembledDiffractionData(
        indexes=indexes_arr,
        patterns=patterns,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        bad_pixels=numpy.zeros((pattern_hw, pattern_hw), dtype=numpy.bool_),
    )


def _product_with_position_indexes(indexes: list[int]) -> Product:
    points = [ProbePosition(index=i, coordinate_x_m=0.0, coordinate_y_m=0.0) for i in indexes]
    return Product(
        metadata=_metadata(),
        probe_positions=ProbePositionSequence(points),
        probes=_make_probes(),
        object_=_make_object(),
        losses=[],
    )


def _product_with_position_specs(specs: list[tuple[int, float, float]]) -> Product:
    """Build a Product from (index, coordinate_x_m, coordinate_y_m) tuples."""
    points = [ProbePosition(index=i, coordinate_x_m=x, coordinate_y_m=y) for i, x, y in specs]
    return Product(
        metadata=_metadata(),
        probe_positions=ProbePositionSequence(points),
        probes=_make_probes(),
        object_=_make_object(),
        losses=[],
    )


def _output_specs(result_product: Product) -> list[tuple[int, float, float]]:
    return [(p.index, p.coordinate_x_m, p.coordinate_y_m) for p in result_product.probe_positions]


class TestPrepareReconstructInputGuards:
    """Guards added so silent reconstruction on empty/inconsistent inputs is impossible."""

    def test_empty_pattern_indexes_raises(self) -> None:
        # All indexes negative -> get_indexes() returns empty after the >=0 filter.
        empty = _make_assembled_data([-1, -1, -1])
        product = _product_with_position_indexes([0, 1, 2])
        with pytest.raises(ValueError, match='empty diffraction dataset'):
            prepare_reconstruct_input(empty, product)

    def test_empty_positions_raises(self) -> None:
        # Zero-length probe_positions: filter result is also empty.
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_indexes([])
        with pytest.raises(ValueError, match='eliminated'):
            prepare_reconstruct_input(data, product)

    def test_pattern_range_above_position_range_raises(self) -> None:
        data = _make_assembled_data([10, 11, 12])
        product = _product_with_position_indexes([0, 1, 2])
        with pytest.raises(ValueError, match='No probe positions overlap'):
            prepare_reconstruct_input(data, product)

    def test_pattern_range_below_position_range_raises(self) -> None:
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_indexes([10, 11, 12])
        with pytest.raises(ValueError, match=r'\[0, 2\].*\[10, 12\]'):
            prepare_reconstruct_input(data, product)

    def test_single_anchor_with_surrounding_patterns_drops_to_match(self) -> None:
        # One unique position at index 5; patterns at [3, 4, 5, 6, 7]. The
        # position-index range collapses to [5, 5], so only pattern index 5
        # survives the in-range trim and it matches exactly. No interpolation
        # is needed, so the operation succeeds with a single output row.
        data = _make_assembled_data([3, 4, 5, 6, 7])
        product = _product_with_position_specs([(5, 1.0e-9, 2.0e-9), (5, 3.0e-9, 4.0e-9)])
        result = prepare_reconstruct_input(data, product)
        out = _output_specs(result.product)
        assert len(out) == 1
        assert out[0][0] == 5
        numpy.testing.assert_allclose(out[0][1], 2.0e-9)
        numpy.testing.assert_allclose(out[0][2], 3.0e-9)
        assert result.diffraction_patterns.shape[0] == 1

    def test_filter_eliminates_all_pattern_indexes_raises(self) -> None:
        # Patterns only at odd indexes; EVEN filter leaves nothing on the pattern axis.
        data = _make_assembled_data([1, 3, 5])
        product = _product_with_position_indexes([0, 2, 4])
        with pytest.raises(ValueError, match='eliminated'):
            prepare_reconstruct_input(data, product, PositionIndexFilter.EVEN)

    def test_filter_eliminates_all_position_indexes_raises(self) -> None:
        # Positions only at odd indexes; EVEN filter leaves nothing on the position axis.
        data = _make_assembled_data([0, 2, 4])
        product = _product_with_position_indexes([1, 3, 5])
        with pytest.raises(ValueError, match='eliminated'):
            prepare_reconstruct_input(data, product, PositionIndexFilter.EVEN)


class TestPrepareReconstructInputMerge:
    """End-to-end behavior of the dedup / interpolate / drop / filter pipeline."""

    # ----- Exact-match / pass-through -----

    def test_full_overlap_passes_through(self) -> None:
        specs = [(i, float(i) * PIXEL_M, -float(i) * PIXEL_M) for i in range(5)]
        data = _make_assembled_data(list(range(5)))
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        assert _output_specs(result.product) == specs
        assert result.diffraction_patterns.shape[0] == 5

    def test_sparse_patterns_dense_positions(self) -> None:
        # Patterns at {0, 2, 4, 6}; positions at {0..6}. Output keeps only the
        # pattern-index subset, with each position's exact coordinates.
        specs = [(i, float(i) * PIXEL_M, 0.0) for i in range(7)]
        data = _make_assembled_data([0, 2, 4, 6])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        expected = [(i, float(i) * PIXEL_M, 0.0) for i in [0, 2, 4, 6]]
        assert _output_specs(result.product) == expected

    def test_pattern_rows_match_pattern_indexes(self) -> None:
        # _make_assembled_data tags each pattern row with its index value. After
        # the merge, each surviving row should still read back its own index.
        data = _make_assembled_data([0, 2, 4, 6])
        product = _product_with_position_indexes(list(range(7)))
        result = prepare_reconstruct_input(data, product)
        row_tags = result.diffraction_patterns[:, 0, 0]
        numpy.testing.assert_array_equal(row_tags, [0, 2, 4, 6])

    # ----- Averaging duplicates -----

    def test_duplicate_position_indexes_are_averaged(self) -> None:
        # Two positions at index 1 with (1, 3) and (3, 5) -> mean is (2, 4).
        specs = [
            (0, 0.0, 0.0),
            (1, 1.0e-9, 3.0e-9),
            (1, 3.0e-9, 5.0e-9),
            (2, 0.0, 0.0),
        ]
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        out = _output_specs(result.product)
        assert out[0] == (0, 0.0, 0.0)
        assert out[1][0] == 1
        numpy.testing.assert_allclose(out[1][1], 2.0e-9)
        numpy.testing.assert_allclose(out[1][2], 4.0e-9)
        assert out[2] == (2, 0.0, 0.0)

    def test_multiple_duplicate_groups_with_different_multiplicities(self) -> None:
        specs = [
            (0, 1.0, 1.0),
            (0, 3.0, 5.0),  # mean (2, 3)
            (1, 10.0, 20.0),  # singleton
            (2, 0.0, 0.0),
            (2, 4.0, 8.0),
            (2, 8.0, 16.0),  # mean (4, 8)
        ]
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        out = _output_specs(result.product)
        assert [o[0] for o in out] == [0, 1, 2]
        numpy.testing.assert_allclose(out[0][1:], (2.0, 3.0))
        numpy.testing.assert_allclose(out[1][1:], (10.0, 20.0))
        numpy.testing.assert_allclose(out[2][1:], (4.0, 8.0))

    def test_duplicates_outside_filter_are_ignored(self) -> None:
        # Even pattern indexes; duplicates at odd index 3 should not influence
        # the EVEN output at all.
        specs = [
            (0, 0.0, 0.0),
            (2, 2.0, 0.0),
            (3, 999.0, 999.0),  # filtered out
            (3, -999.0, -999.0),  # filtered out
            (4, 4.0, 0.0),
        ]
        data = _make_assembled_data([0, 2, 4])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product, PositionIndexFilter.EVEN)
        assert _output_specs(result.product) == [(0, 0.0, 0.0), (2, 2.0, 0.0), (4, 4.0, 0.0)]

    # ----- Interpolation -----

    def test_single_missing_index_is_linearly_interpolated(self) -> None:
        # Positions at [0, 2]; pattern at [0, 1, 2]. Output index 1 is the midpoint.
        specs = [(0, 0.0, 0.0), (2, 10.0, -6.0)]
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        out = _output_specs(result.product)
        assert [o[0] for o in out] == [0, 1, 2]
        numpy.testing.assert_allclose(out[0][1:], (0.0, 0.0))
        numpy.testing.assert_allclose(out[1][1:], (5.0, -3.0))
        numpy.testing.assert_allclose(out[2][1:], (10.0, -6.0))

    def test_consecutive_missing_indexes_are_linearly_interpolated(self) -> None:
        # Positions at [0, 5]; pattern at [0..5]. Each gap is 1/5 of the span.
        specs = [(0, 0.0, 0.0), (5, 5.0, 10.0)]
        data = _make_assembled_data(list(range(6)))
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        out = _output_specs(result.product)
        assert [o[0] for o in out] == [0, 1, 2, 3, 4, 5]
        xs = [o[1] for o in out]
        ys = [o[2] for o in out]
        numpy.testing.assert_allclose(xs, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        numpy.testing.assert_allclose(ys, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

    def test_interp_uses_averaged_duplicates_as_anchors(self) -> None:
        # Duplicates at 0 average to (2, 0); duplicates at 4 average to (10, 20).
        # Pattern indexes 1, 2, 3 land at quarter, half, three-quarter intervals.
        specs = [
            (0, 1.0, -1.0),
            (0, 3.0, 1.0),  # mean (2, 0)
            (4, 8.0, 18.0),
            (4, 12.0, 22.0),  # mean (10, 20)
        ]
        data = _make_assembled_data([0, 1, 2, 3, 4])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        xs = [p.coordinate_x_m for p in result.product.probe_positions]
        ys = [p.coordinate_y_m for p in result.product.probe_positions]
        numpy.testing.assert_allclose(xs, [2.0, 4.0, 6.0, 8.0, 10.0])
        numpy.testing.assert_allclose(ys, [0.0, 5.0, 10.0, 15.0, 20.0])

    def test_exact_match_indexes_are_not_modified(self) -> None:
        # Positions at every pattern index, with deliberately non-linear coords.
        # Exact-match should pass through verbatim (no smoothing).
        specs = [(0, 1.0, -7.0), (1, 100.0, 0.5), (2, -3.0, 42.0)]
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        assert _output_specs(result.product) == specs

    # ----- Out-of-range drop -----

    def test_pattern_indexes_above_range_dropped(self) -> None:
        specs = [(0, 0.0, 0.0), (1, 1.0, 0.0), (2, 2.0, 0.0)]
        data = _make_assembled_data([0, 1, 2, 3, 4])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        assert [p.index for p in result.product.probe_positions] == [0, 1, 2]

    def test_pattern_indexes_below_range_dropped(self) -> None:
        specs = [(3, 3.0, 0.0), (4, 4.0, 0.0), (5, 5.0, 0.0)]
        data = _make_assembled_data([0, 1, 2, 3, 4, 5])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        assert [p.index for p in result.product.probe_positions] == [3, 4, 5]

    def test_pattern_indexes_straddling_both_ends_dropped(self) -> None:
        specs = [(2, 2.0, 0.0), (3, 3.0, 0.0), (4, 4.0, 0.0)]
        data = _make_assembled_data([0, 1, 2, 3, 4, 5, 6])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        assert [p.index for p in result.product.probe_positions] == [2, 3, 4]

    def test_boundary_pattern_indexes_inclusive(self) -> None:
        # Exactly lo and exactly hi survive trimming.
        specs = [(2, 2.0, 0.0), (5, 5.0, 0.0)]
        data = _make_assembled_data([2, 5])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        assert _output_specs(result.product) == specs

    # ----- Extra positions dropped (no-op) -----

    def test_positions_with_no_pattern_are_not_in_output(self) -> None:
        specs = [(i, float(i), 0.0) for i in range(10)]
        data = _make_assembled_data([0, 5])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        assert _output_specs(result.product) == [(0, 0.0, 0.0), (5, 5.0, 0.0)]
        assert result.diffraction_patterns.shape[0] == 2

    # ----- Filters re-validated -----

    def test_filter_even_full_overlap(self) -> None:
        data = _make_assembled_data(list(range(10)))
        product = _product_with_position_indexes(list(range(10)))
        result = prepare_reconstruct_input(data, product, PositionIndexFilter.EVEN)
        assert [p.index for p in result.product.probe_positions] == [0, 2, 4, 6, 8]

    def test_filter_odd_full_overlap(self) -> None:
        data = _make_assembled_data(list(range(10)))
        product = _product_with_position_indexes(list(range(10)))
        result = prepare_reconstruct_input(data, product, PositionIndexFilter.ODD)
        assert [p.index for p in result.product.probe_positions] == [1, 3, 5, 7, 9]

    def test_filter_odd_with_duplicates_and_interp(self) -> None:
        # ODD filter: keep odd indexes on both axes. Positions at odd indexes
        # 1 (duplicated, mean coord = (2, 0)), 3, 7. Patterns at [1, 3, 5, 7].
        # Pattern 5 is missing from positions -> interpolated between 3 (x=4)
        # and 7 (x=8) at fraction (5-3)/(7-3) = 0.5 -> x = 6.
        specs = [
            (1, 1.0, 0.0),
            (1, 3.0, 0.0),  # averaged to (2, 0)
            (2, -100.0, -100.0),  # filtered out
            (3, 4.0, 0.0),
            (7, 8.0, 0.0),
            (8, -100.0, -100.0),  # filtered out
        ]
        data = _make_assembled_data([1, 3, 5, 7])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product, PositionIndexFilter.ODD)
        out = _output_specs(result.product)
        assert [o[0] for o in out] == [1, 3, 5, 7]
        numpy.testing.assert_allclose([o[1] for o in out], [2.0, 4.0, 6.0, 8.0])
        numpy.testing.assert_allclose([o[2] for o in out], [0.0, 0.0, 0.0, 0.0])

    # ----- End-to-end combined -----

    def test_dedup_interp_and_drop_combined(self) -> None:
        # Pattern indexes [0, 2, 4, 6, 8, 10]. Position indexes (after dedup):
        #   2 duplicated -> mean (2, 0)
        #   6 single anchor (6, 0)
        #   8 single anchor (8, 0)
        # Position range = [2, 8]. Pattern 0 (below) and 10 (above) are dropped.
        # Pattern 4 is interpolated between 2 (x=2) and 6 (x=6) -> x=4.
        # Patterns 2, 6, 8 match exactly.
        specs = [
            (2, 1.0, 0.0),
            (2, 3.0, 0.0),  # averaged to (2, 0)
            (6, 6.0, 0.0),
            (8, 8.0, 0.0),
        ]
        data = _make_assembled_data([0, 2, 4, 6, 8, 10])
        product = _product_with_position_specs(specs)
        result = prepare_reconstruct_input(data, product)
        out = _output_specs(result.product)
        assert [o[0] for o in out] == [2, 4, 6, 8]
        numpy.testing.assert_allclose([o[1] for o in out], [2.0, 4.0, 6.0, 8.0])
        numpy.testing.assert_allclose([o[2] for o in out], [0.0, 0.0, 0.0, 0.0])
        # Pattern rows are tagged with their index; confirm we sliced correctly.
        row_tags = result.diffraction_patterns[:, 0, 0]
        numpy.testing.assert_array_equal(row_tags, [2, 4, 6, 8])

    def test_product_metadata_probes_object_passed_through(self) -> None:
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_indexes([0, 1, 2])
        result = prepare_reconstruct_input(data, product)
        assert result.product.metadata is product.metadata
        assert result.product.probes is product.probes
        assert result.product.object_ is product.object_
        assert result.product.losses is product.losses

    def test_bad_pixels_passed_through_unchanged(self) -> None:
        data = _make_assembled_data([0, 1, 2])
        product = _product_with_position_indexes([0, 1, 2])
        result = prepare_reconstruct_input(data, product)
        assert result.bad_pixels is data.get_bad_pixels()


class TestNullReconstructor:
    """NullReconstructor must remain instantiable as new abstract methods are added."""

    def test_instantiable(self) -> None:
        # Will raise TypeError if any abstract method on TrainableReconstructor lacks a stub.
        NullReconstructor('null')

    def test_get_model_file_extension_returns_empty(self) -> None:
        assert NullReconstructor('null').get_model_file_extension() == ''

    def test_save_model_is_noop(self, tmp_path: Path) -> None:
        target = tmp_path / 'model.bin'
        NullReconstructor('null').save_model(target)
        assert not target.exists()
