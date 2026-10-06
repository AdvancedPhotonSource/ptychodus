"""Unit tests for probe generation functions in ptychodus.api.simulate.probe."""

from typing import Any
import math

import numpy
import numpy.testing
import pytest
from pydantic import ValidationError

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import HermiteMode, ImageExtent, LegendreMode, PixelGeometry
from ptychodus.api.probe import Probe, ProbeGeometry
from ptychodus.api.simulate.probe import (
    propagate_probe,
    DEFAULT_INCOHERENT_MODE_STRATEGY,
    FresnelZonePlate,
    ProbeModeDecayType,
    ProbeMomentPolynomialStrategy,
    _gram_schmidt,
    GaussianSchellStrategy,
    IncoherentModeStrategy,
    KirkpatrickBaezMirror,
    KirkpatrickBaezMirrorPair,
    generate_kb_mirror_probe,
    generate_mirror_figure_error,
    generate_average_pattern_probe,
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_hermite_probe,
    RandomPhaseRampStrategy,
    generate_incoherent_probe_modes,
)
from ptychodus.api.propagate import (
    AngularSpectrumPropagator,
    FresnelTransformPropagator,
    compute_far_field_pixel_geometry,
    PropagatorParameters,
    intensity,
)


PIXEL_GEOMETRY = PixelGeometry(width_m=1e-8, height_m=1e-8)


def _make_single_mode_probe(height: int = 16, width: int = 16, seed: int = 0) -> Probe:
    rng = numpy.random.default_rng(seed)
    array = rng.standard_normal((1, height, width)) + 1j * rng.standard_normal((1, height, width))
    return Probe(array=array, pixel_geometry=PIXEL_GEOMETRY)


class TestGenerateIncoherentProbeModes:
    def test_orthogonalization_applied_when_expanding_from_one_mode(self) -> None:
        """Regression test: orthogonalization must run even when the input has only 1 mode.

        Before the fix, the guard was ``array_in.shape[-3] > 1`` which is always False
        when starting from a single-mode probe, so orthogonalization was silently skipped.
        """
        rng = numpy.random.default_rng(42)
        probe = _make_single_mode_probe()
        num_modes = 4

        result = generate_incoherent_probe_modes(
            probe, num_modes, strategy=RandomPhaseRampStrategy(rng), orthogonalize=True
        )

        array = result.get_array()
        assert array.shape[0] == num_modes

        # Flatten each mode to a row and check pairwise inner products ≈ 0
        modes = array.reshape(num_modes, -1)
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                dot = numpy.abs(numpy.vdot(modes[i], modes[j]))
                norm_i = numpy.linalg.norm(modes[i])
                norm_j = numpy.linalg.norm(modes[j])
                # Normalise so the tolerance is scale-independent
                assert dot / (norm_i * norm_j) < 1e-10, (
                    f'Modes {i} and {j} are not orthogonal (|<i|j>|/(||i||·||j||) = {dot / (norm_i * norm_j):.3e})'
                )

    def test_orthogonalization_skipped_when_disabled(self) -> None:
        """With orthogonalize=False the output modes need not be orthogonal."""
        rng = numpy.random.default_rng(0)
        probe = _make_single_mode_probe()

        result = generate_incoherent_probe_modes(
            probe, 3, strategy=RandomPhaseRampStrategy(rng), orthogonalize=False
        )

        # Just verify shape and no NaNs — orthogonality is NOT required here.
        array = result.get_array()
        assert array.shape[0] == 3
        assert not numpy.isnan(array).any()

    def test_single_output_mode_unchanged(self) -> None:
        """A single-element weight list should return a probe with one mode."""
        rng = numpy.random.default_rng(7)
        probe = _make_single_mode_probe()

        result = generate_incoherent_probe_modes(
            probe, 1, strategy=RandomPhaseRampStrategy(rng), orthogonalize=True
        )

        assert result.get_array().shape[0] == 1

    def test_intensity_weights_respected(self) -> None:
        """Output mode intensities should be proportional to the requested weights."""
        rng = numpy.random.default_rng(99)
        probe = _make_single_mode_probe()
        strategy = RandomPhaseRampStrategy(rng, decay_ratio=0.5)

        result = generate_incoherent_probe_modes(probe, 3, strategy=strategy, orthogonalize=True)

        array = result.get_array()
        intensities = numpy.array([numpy.sum(numpy.abs(array[m]) ** 2) for m in range(3)])
        ratios = intensities / intensities[0]
        expected = numpy.array([1.0, 0.5, 0.25])
        numpy.testing.assert_allclose(ratios, expected, rtol=1e-6)


class TestGenerateCoherentProbeModes:
    def _make_multimode_probe(self, num_imodes: int = 5, seed: int = 3) -> Probe:
        rng = numpy.random.default_rng(seed)
        single = _make_single_mode_probe(seed=seed)
        strategy = RandomPhaseRampStrategy(rng, decay_ratio=1.0)
        return generate_incoherent_probe_modes(
            single, num_imodes, strategy=strategy, orthogonalize=True
        )

    def test_eigenmode_incoherent_slots_are_all_nonzero(self) -> None:
        """Regression test: eigenmodes must fill every incoherent mode.

        Before the fix only incoherent slot 0 of each eigenmode was populated,
        leaving zero-power slots that trigger a 0/0 -> NaN in a downstream
        Gram-Schmidt orthogonalization (segfault/abort on a GPU backend).
        """
        rng = numpy.random.default_rng(11)
        num_imodes = 5
        num_cmodes = 2
        probe = self._make_multimode_probe(num_imodes=num_imodes)

        result = generate_coherent_probe_modes(
            rng, probe, num_cmodes=num_cmodes, num_diffraction_patterns=8
        )

        array = result.get_array()
        assert array.shape == (num_cmodes, num_imodes, probe.height_px, probe.width_px)
        assert not numpy.isnan(array).any()

        for cmode in range(1, num_cmodes):
            for imode in range(num_imodes):
                power = numpy.sum(intensity(array[cmode, imode]))
                assert power > 0.0, f'eigenmode ({cmode}, {imode}) has zero power'

    def test_opr_weights_shape_and_main_column(self) -> None:
        rng = numpy.random.default_rng(12)
        probe = self._make_multimode_probe(num_imodes=3)

        result = generate_coherent_probe_modes(rng, probe, num_cmodes=2, num_diffraction_patterns=8)

        weights = result.get_opr_weights()
        assert weights.shape == (8, 2)
        numpy.testing.assert_array_equal(weights[:, 0], numpy.ones(8))

    def test_single_coherent_mode_has_no_opr_weights(self) -> None:
        rng = numpy.random.default_rng(13)
        probe = self._make_multimode_probe(num_imodes=3)

        result = generate_coherent_probe_modes(rng, probe, num_cmodes=1, num_diffraction_patterns=8)

        assert result.get_array().shape[0] == 1
        assert result.get_opr_weights_or_none() is None


def _probe_geometry(height_px: int = 16, width_px: int = 16) -> ProbeGeometry:
    return ProbeGeometry(
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=PIXEL_GEOMETRY.width_m,
        pixel_height_m=PIXEL_GEOMETRY.height_m,
    )


class TestGenerateHermiteProbe:
    def test_returns_probe_with_input_pixel_geometry(self) -> None:
        geometry = _probe_geometry()
        result = generate_hermite_probe(
            geometry, [HermiteMode(1.0, 0, 0)], width_m=1e-7, height_m=1e-7
        )
        assert result.get_pixel_geometry() == geometry.get_pixel_geometry()

    def test_returns_probe_with_geometry_shape(self) -> None:
        geometry = _probe_geometry(height_px=12, width_px=20)
        result = generate_hermite_probe(
            geometry, [HermiteMode(1.0, 1, 2)], width_m=1e-7, height_m=1e-7
        )
        array = result.get_array()
        # Probe packs incoherent modes in a leading dim: (num_modes, height, width).
        assert array.shape[-2:] == (geometry.height_px, geometry.width_px)
        assert numpy.iscomplexobj(array)

    def test_empty_modes_returns_zero_probe(self) -> None:
        geometry = _probe_geometry()
        result = generate_hermite_probe(geometry, [], width_m=1e-7, height_m=1e-7)
        array = result.get_array()
        assert array.shape[-2:] == (geometry.height_px, geometry.width_px)
        numpy.testing.assert_array_equal(array, numpy.zeros_like(array))

    def test_piston_mode_is_constant(self) -> None:
        """A single HermiteMode(c, 0, 0) yields an array of constant value c."""
        geometry = _probe_geometry()
        coefficient = 1.7 - 0.3j
        for scale_m in (1e-8, 1.0, 1e3):
            result = generate_hermite_probe(
                geometry, [HermiteMode(coefficient, 0, 0)], width_m=scale_m, height_m=scale_m
            )
            numpy.testing.assert_allclose(result.get_array(), coefficient)

    def test_linearity_over_modes(self) -> None:
        """Sum of probes from individual modes equals the probe from the combined list."""
        geometry = _probe_geometry()
        scale_m = 2e-7
        modes = [HermiteMode(2.0 + 1j, 1, 0), HermiteMode(-0.5j, 0, 2), HermiteMode(0.7, 2, 1)]

        combined = generate_hermite_probe(
            geometry, modes, width_m=scale_m, height_m=scale_m
        ).get_array()
        separate = sum(
            generate_hermite_probe(geometry, [m], width_m=scale_m, height_m=scale_m).get_array()
            for m in modes
        )
        numpy.testing.assert_allclose(combined, separate, atol=1e-12)

    def test_width_inversely_scales_x_argument(self) -> None:
        """For H_1(x) = 2x/width_m, doubling width_m halves the returned array."""
        geometry = _probe_geometry()
        mode = HermiteMode(1.0, 1, 0)
        scale_m = 1e-7
        small = generate_hermite_probe(
            geometry, [mode], width_m=scale_m, height_m=scale_m
        ).get_array()
        large = generate_hermite_probe(
            geometry, [mode], width_m=2.0 * scale_m, height_m=scale_m
        ).get_array()
        numpy.testing.assert_allclose(large, 0.5 * small, atol=1e-12)

    def test_independent_axis_scaling(self) -> None:
        """Doubling height_m halves the y-argument; x-argument is unaffected."""
        geometry = _probe_geometry()
        mode_y = HermiteMode(1.0, 0, 1)  # H_1(y) = 2y
        small = generate_hermite_probe(geometry, [mode_y], width_m=1e-7, height_m=1e-7).get_array()
        large = generate_hermite_probe(geometry, [mode_y], width_m=1e-7, height_m=2e-7).get_array()
        numpy.testing.assert_allclose(large, 0.5 * small, atol=1e-12)


# ---------------------------------------------------------------------------
# generate_average_pattern_probe
# ---------------------------------------------------------------------------

_BACKPROP_NUM_PX = 64
_BACKPROP_WAVELENGTH_M = 1.24e-10
_BACKPROP_DISTANCE_M = 1.0
_BACKPROP_DETECTOR_PITCH_M = 75e-6
_BACKPROP_SAMPLE_PITCH_M = (
    _BACKPROP_WAVELENGTH_M * _BACKPROP_DISTANCE_M / (_BACKPROP_NUM_PX * _BACKPROP_DETECTOR_PITCH_M)
)


def _assembled_data(patterns: numpy.ndarray) -> AssembledDiffractionData:
    num_patterns, height_px, width_px = patterns.shape
    return AssembledDiffractionData(
        indexes=numpy.arange(num_patterns),
        patterns=patterns,
        pixel_geometry=PixelGeometry(
            width_m=_BACKPROP_DETECTOR_PITCH_M, height_m=_BACKPROP_DETECTOR_PITCH_M
        ),
        bad_pixels=numpy.zeros((height_px, width_px), dtype=bool),
    )


def _sample_plane_geometry() -> ProbeGeometry:
    return ProbeGeometry(
        width_px=_BACKPROP_NUM_PX,
        height_px=_BACKPROP_NUM_PX,
        pixel_width_m=_BACKPROP_SAMPLE_PITCH_M,
        pixel_height_m=_BACKPROP_SAMPLE_PITCH_M,
    )


class TestGenerateAveragePatternProbe:
    def test_conserves_energy_between_the_two_planes(self) -> None:
        """Power is invariant across the propagation once each plane is weighted by its
        own pixel area: ``sum|out|^2 dx_sample^2 == sum|in|^2 dx_detector^2``.

        This pins the upstream-plane pitch convention without needing phase, so it holds
        for the sqrt-of-intensity input the estimator actually takes. The single-FFT
        propagator's ``pixel_width_m`` names the plane at the smaller z, which for this
        backward propagation is the *sample* plane. Parameterizing it with the detector
        pitch instead scales the recovered power by ``(dx_sample / dx_detector)**4``
        -- here a factor of about 1.4e-14.
        """
        geometry = _sample_plane_geometry()
        rng = numpy.random.default_rng(0)
        coord = numpy.arange(_BACKPROP_NUM_PX) - _BACKPROP_NUM_PX // 2
        envelope = numpy.exp(-(coord[:, None] ** 2 + coord[None, :] ** 2) / 100)
        patterns = (envelope * rng.uniform(0.5, 1.5, envelope.shape))[numpy.newaxis]

        result = generate_average_pattern_probe(
            geometry,
            _assembled_data(patterns),
            probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
            detector_distance_m=_BACKPROP_DISTANCE_M,
        )

        detector_power = patterns[0].sum() * _BACKPROP_DETECTOR_PITCH_M**2
        sample_power = intensity(result.get_array()[0]).sum() * _BACKPROP_SAMPLE_PITCH_M**2
        assert sample_power == pytest.approx(detector_power, rel=1e-12)

    def test_recovers_the_forward_models_amplitude_scale(self) -> None:
        """Round-trip through the house forward model (sample -> detector at +z) and
        back. Phase is lost to the sqrt-of-intensity step so the field itself cannot
        return, but the recovered probe must carry the same total power as the original
        -- which is false by four orders of magnitude on the wrong grid.
        """
        geometry = _sample_plane_geometry()
        coord = numpy.arange(_BACKPROP_NUM_PX) - _BACKPROP_NUM_PX // 2
        probe = numpy.exp(-(coord[:, None] ** 2 + coord[None, :] ** 2) / 50).astype(complex)

        forward = FresnelTransformPropagator(
            PropagatorParameters(
                wavelength_m=_BACKPROP_WAVELENGTH_M,
                width_px=_BACKPROP_NUM_PX,
                height_px=_BACKPROP_NUM_PX,
                pixel_width_m=_BACKPROP_SAMPLE_PITCH_M,
                pixel_height_m=_BACKPROP_SAMPLE_PITCH_M,
                propagation_distance_m=_BACKPROP_DISTANCE_M,
            )
        )
        patterns = intensity(forward.propagate(probe))[numpy.newaxis]

        result = generate_average_pattern_probe(
            geometry,
            _assembled_data(patterns),
            probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
            detector_distance_m=_BACKPROP_DISTANCE_M,
        )

        expected_power = intensity(probe).sum()
        recovered_power = intensity(result.get_array()[0]).sum()
        assert recovered_power == pytest.approx(expected_power, rel=1e-9)

    def test_returns_the_declared_sample_plane_geometry(self) -> None:
        geometry = _sample_plane_geometry()
        patterns = numpy.ones((2, _BACKPROP_NUM_PX, _BACKPROP_NUM_PX))
        result = generate_average_pattern_probe(
            geometry,
            _assembled_data(patterns),
            probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
            detector_distance_m=_BACKPROP_DISTANCE_M,
        )
        assert result.get_pixel_geometry() == geometry.get_pixel_geometry()
        assert result.get_array().shape[-2:] == (_BACKPROP_NUM_PX, _BACKPROP_NUM_PX)

    def test_zero_detector_distance_raises(self) -> None:
        """Previously returned an all-NaN probe in silence: at z=0 the implied and
        declared pitches are both 0, so the existing isclose() guard passed."""
        patterns = numpy.ones((1, _BACKPROP_NUM_PX, _BACKPROP_NUM_PX))
        with pytest.raises(ValueError, match='Detector distance must be nonzero'):
            generate_average_pattern_probe(
                _sample_plane_geometry(),
                _assembled_data(patterns),
                probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
                detector_distance_m=0.0,
            )


# A near-field product samples the object on the detector grid itself, so the
# "sample pitch" is the detector pitch demagnified rather than the reciprocal relation.
_NEAR_FIELD_MAGNIFICATION = 4.0
_NEAR_FIELD_SAMPLE_PITCH_M = _BACKPROP_DETECTOR_PITCH_M / _NEAR_FIELD_MAGNIFICATION


def _near_field_sample_geometry(magnification: float = _NEAR_FIELD_MAGNIFICATION) -> ProbeGeometry:
    pitch_m = _BACKPROP_DETECTOR_PITCH_M / magnification
    return ProbeGeometry(
        width_px=_BACKPROP_NUM_PX,
        height_px=_BACKPROP_NUM_PX,
        pixel_width_m=pitch_m,
        pixel_height_m=pitch_m,
    )


class TestGenerateAveragePatternProbeNearField:
    """Near field back-propagates with the angular spectrum, which preserves pitch, so
    the probe lands on the grid the caller already declared and there is no output
    pitch to reconcile.
    """

    def test_returns_the_declared_sample_pitch(self) -> None:
        patterns = numpy.ones((1, _BACKPROP_NUM_PX, _BACKPROP_NUM_PX))
        geometry = _near_field_sample_geometry()

        result = generate_average_pattern_probe(
            geometry,
            _assembled_data(patterns),
            probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
            detector_distance_m=_BACKPROP_DISTANCE_M,
            far_field=False,
        )

        assert result.get_pixel_geometry() == geometry.get_pixel_geometry()
        assert result.get_array().shape[-2:] == (_BACKPROP_NUM_PX, _BACKPROP_NUM_PX)

    def test_the_far_field_pitch_check_does_not_fire(self) -> None:
        """A near-field grid disagrees with the Fraunhofer relation by orders of
        magnitude; before near field was supported this raised, which left the
        estimator unusable on exactly the products it suits best.
        """
        patterns = numpy.ones((1, _BACKPROP_NUM_PX, _BACKPROP_NUM_PX))

        assert _NEAR_FIELD_SAMPLE_PITCH_M != pytest.approx(_BACKPROP_SAMPLE_PITCH_M)
        generate_average_pattern_probe(
            _near_field_sample_geometry(),
            _assembled_data(patterns),
            probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
            detector_distance_m=_BACKPROP_DISTANCE_M,
            far_field=False,
        )

    def test_matches_a_directly_constructed_angular_spectrum_propagator(self) -> None:
        """The magnification is read off the pitch ratio, which fixes the equivalent
        parallel-beam distance at z_d / M."""
        rng = numpy.random.default_rng(7)
        patterns = rng.random((3, _BACKPROP_NUM_PX, _BACKPROP_NUM_PX))
        geometry = _near_field_sample_geometry()

        result = generate_average_pattern_probe(
            geometry,
            _assembled_data(patterns),
            probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
            detector_distance_m=_BACKPROP_DISTANCE_M,
            far_field=False,
        )

        params = PropagatorParameters(
            wavelength_m=_BACKPROP_WAVELENGTH_M,
            width_px=_BACKPROP_NUM_PX,
            height_px=_BACKPROP_NUM_PX,
            pixel_width_m=_NEAR_FIELD_SAMPLE_PITCH_M,
            pixel_height_m=_NEAR_FIELD_SAMPLE_PITCH_M,
            propagation_distance_m=-_BACKPROP_DISTANCE_M / _NEAR_FIELD_MAGNIFICATION,
        )
        expected = AngularSpectrumPropagator(params).propagate(
            numpy.sqrt(numpy.mean(patterns, axis=0)).astype(complex)
        )

        # get_array normalizes the single mode to a (1, H, W) stack.
        numpy.testing.assert_allclose(result.get_array()[0], expected, rtol=1e-12)

    def test_the_far_field_path_is_unchanged_by_default(self) -> None:
        """far_field defaults to True, so existing callers keep the Fraunhofer check."""
        patterns = numpy.ones((1, _BACKPROP_NUM_PX, _BACKPROP_NUM_PX))

        with pytest.raises(ValueError, match='does not match'):
            generate_average_pattern_probe(
                _near_field_sample_geometry(),
                _assembled_data(patterns),
                probe_wavelength_m=_BACKPROP_WAVELENGTH_M,
                detector_distance_m=_BACKPROP_DISTANCE_M,
            )


_FZP_NUM_PX = 64
_FZP_WAVELENGTH_M = 1.24e-10
_FZP_PROBE_PITCH_M = 8e-9
_FZP_ZONE_PLATE = FresnelZonePlate(
    zone_plate_diameter_m=180e-6,
    outermost_zone_width_m=50e-9,
    central_beamstop_diameter_m=60e-6,
)
_FZP_FOCAL_LENGTH_M = _FZP_ZONE_PLATE.get_focal_length_m(_FZP_WAVELENGTH_M)


def _fzp_probe_geometry() -> ProbeGeometry:
    return ProbeGeometry(
        width_px=_FZP_NUM_PX,
        height_px=_FZP_NUM_PX,
        pixel_width_m=_FZP_PROBE_PITCH_M,
        pixel_height_m=_FZP_PROBE_PITCH_M,
    )


def _fzp_plane_pixel_geometry() -> PixelGeometry:
    """Pitch of the zone-plate plane, conjugate to the probe plane across ``|z| = f``.

    Both the forward (``defocus = 0``) and backward (``defocus = -2f``) cases run over the
    same ``|z| = f``, so they share this grid.
    """
    return compute_far_field_pixel_geometry(
        PixelGeometry(width_m=_FZP_PROBE_PITCH_M, height_m=_FZP_PROBE_PITCH_M),
        ImageExtent(width_px=_FZP_NUM_PX, height_px=_FZP_NUM_PX),
        wavelength_m=_FZP_WAVELENGTH_M,
        propagation_distance_m=_FZP_FOCAL_LENGTH_M,
    )


def _zone_plate_annulus() -> numpy.ndarray:
    """Boolean mask of the zone plate's open area, sampled on the FZP-plane grid."""
    fzp_pixel_geometry = _fzp_plane_pixel_geometry()
    coord = numpy.arange(_FZP_NUM_PX) - _FZP_NUM_PX // 2
    radius_m = numpy.hypot(
        fzp_pixel_geometry.width_m * coord[numpy.newaxis, :],
        fzp_pixel_geometry.height_m * coord[:, numpy.newaxis],
    )
    return (radius_m <= _FZP_ZONE_PLATE.zone_plate_diameter_m / 2) & (
        radius_m >= _FZP_ZONE_PLATE.central_beamstop_diameter_m / 2
    )


def _invert_onto_the_zone_plate_plane(
    probe_plane_array: numpy.ndarray, *, defocus_distance_m: float
) -> numpy.ndarray:
    """Undo the propagation the generator applied, returning the field to the FZP plane.

    The generator applies ``P(z, dx_upstream)`` where ``z = f + defocus`` and the upstream
    plane is the FZP for ``z > 0`` and the probe plane for ``z < 0``. The inverse of
    ``P(z, dx)`` is ``P(-z, dx)`` -- same pitch, negated distance -- which is exactly the
    direction semantics under test.
    """
    propagation_distance_m = _FZP_FOCAL_LENGTH_M + defocus_distance_m
    upstream_pixel_geometry = (
        _fzp_plane_pixel_geometry()
        if propagation_distance_m > 0.0
        else PixelGeometry(width_m=_FZP_PROBE_PITCH_M, height_m=_FZP_PROBE_PITCH_M)
    )
    propagator = FresnelTransformPropagator(
        PropagatorParameters(
            wavelength_m=_FZP_WAVELENGTH_M,
            width_px=_FZP_NUM_PX,
            height_px=_FZP_NUM_PX,
            pixel_width_m=upstream_pixel_geometry.width_m,
            pixel_height_m=upstream_pixel_geometry.height_m,
            propagation_distance_m=-propagation_distance_m,
        )
    )
    return propagator.propagate(probe_plane_array)


class TestFresnelZonePlate:
    """The zone plate carried no validation at all before it became a model."""

    def _sound(self) -> dict:
        return dict(
            zone_plate_diameter_m=180e-6,
            outermost_zone_width_m=50e-9,
            central_beamstop_diameter_m=60e-6,
        )

    @pytest.mark.parametrize(
        'override',
        [
            {'zone_plate_diameter_m': 0.0},
            {'zone_plate_diameter_m': -1e-6},
            {'zone_plate_diameter_m': math.inf},
            {'outermost_zone_width_m': 0.0},
            {'outermost_zone_width_m': -1e-9},
            {'central_beamstop_diameter_m': -1e-6},
        ],
    )
    def test_rejects_a_size_that_is_not_a_size(self, override: dict) -> None:
        kwargs = self._sound()
        kwargs.update(override)

        with pytest.raises(ValidationError, match=next(iter(override))):
            FresnelZonePlate(**kwargs)

    def test_a_zero_beamstop_means_no_central_stop(self) -> None:
        """Zero is a legitimate value: the pupil is then the full disk."""
        zone_plate = FresnelZonePlate(**{**self._sound(), 'central_beamstop_diameter_m': 0.0})

        assert zone_plate.central_beamstop_diameter_m == 0.0

    @pytest.mark.parametrize('beamstop_m', [180e-6, 200e-6])
    def test_rejects_a_beamstop_that_swallows_the_aperture(self, beamstop_m: float) -> None:
        """The pupil is the annulus between stop and rim, so a stop that reaches the rim
        leaves nothing; that used to produce a silently black probe."""
        kwargs = {**self._sound(), 'central_beamstop_diameter_m': beamstop_m}

        with pytest.raises(ValidationError, match='smaller than the zone plate diameter'):
            FresnelZonePlate(**kwargs)

    def test_rejects_an_unknown_field(self) -> None:
        """A preset is a bare literal call, so a misspelled field must not pass silently."""
        with pytest.raises(ValidationError, match='zoneplate_diameter_m'):
            # Misspelled on purpose; mypy agreeing the field is unknown is the point.
            FresnelZonePlate(**self._sound(), zoneplate_diameter_m=1.0)  # type: ignore[call-arg]


class TestGenerateFresnelZonePlateProbe:
    """The sign of ``focal_length + defocus`` selects the propagation direction.

    ``defocus = 0`` puts the probe one focal length downstream of the zone plate (forward,
    ``z = +f``); ``defocus = -2f`` puts it one focal length upstream (backward, ``z = -f``).
    Both run over the same ``|z|``, so they share a zone-plate-plane grid and differ only
    in direction -- which is what makes them comparable.
    """

    @pytest.mark.parametrize('defocus_distance_m', [0.0, -2 * _FZP_FOCAL_LENGTH_M])
    def test_declares_the_probe_plane_geometry_in_both_directions(
        self, defocus_distance_m: float
    ) -> None:
        geometry = _fzp_probe_geometry()

        probe = generate_fresnel_zone_plate_probe(
            geometry,
            _FZP_ZONE_PLATE,
            probe_wavelength_m=_FZP_WAVELENGTH_M,
            defocus_distance_m=defocus_distance_m,
        )

        assert probe.get_array().shape == (1, _FZP_NUM_PX, _FZP_NUM_PX)
        assert probe.get_pixel_geometry() == geometry.get_pixel_geometry()
        assert numpy.all(numpy.isfinite(probe.get_array()))

    def test_both_directions_invert_to_the_same_transmission_function(self) -> None:
        """The end-to-end statement of the sign convention.

        Forward and backward apply different operators to the *same* zone-plate
        transmission function. Undoing each with its own inverse must therefore land on
        one field -- the transmission function itself. On the shipped tree the backward
        branch was parameterized with the zone-plate pitch (and an unsigned distance), so
        its inverse lands somewhere else entirely.
        """
        geometry = _fzp_probe_geometry()
        recovered = [
            _invert_onto_the_zone_plate_plane(
                generate_fresnel_zone_plate_probe(
                    geometry,
                    _FZP_ZONE_PLATE,
                    probe_wavelength_m=_FZP_WAVELENGTH_M,
                    defocus_distance_m=defocus_distance_m,
                ).get_array()[0],
                defocus_distance_m=defocus_distance_m,
            )
            for defocus_distance_m in (0.0, -2 * _FZP_FOCAL_LENGTH_M)
        ]

        numpy.testing.assert_allclose(recovered[1], recovered[0], atol=1e-12)

        # Guard against both inversions agreeing on something that is not the zone plate:
        # the recovered field must be a unit-modulus annulus on the FZP grid.
        annulus = _zone_plate_annulus()
        assert annulus.any() and not annulus.all(), 'Test precondition: grid straddles the optic'
        numpy.testing.assert_allclose(numpy.abs(recovered[0][annulus]), 1.0, atol=1e-12)
        numpy.testing.assert_allclose(numpy.abs(recovered[0][~annulus]), 0.0, atol=1e-12)

    @pytest.mark.parametrize('defocus_distance_m', [0.0, -2 * _FZP_FOCAL_LENGTH_M])
    def test_conserves_energy_between_the_zone_plate_and_probe_planes(
        self, defocus_distance_m: float
    ) -> None:
        """Pins the grid each direction actually outputs on, independent of phase.

        The zone plate's open area is fixed by the optic: the transmission function is
        unit-modulus on the annulus and zero elsewhere, so its power is just that area.
        The generator promises the probe plane's pitch in both directions, so the probe's
        area-weighted power must equal it. Parameterizing the backward branch on the
        zone-plate pitch instead puts the output on a grid coarser by
        ``(dx_fzp / dx_probe)**2``, and the powers part company.
        """
        geometry = _fzp_probe_geometry()

        probe = generate_fresnel_zone_plate_probe(
            geometry,
            _FZP_ZONE_PLATE,
            probe_wavelength_m=_FZP_WAVELENGTH_M,
            defocus_distance_m=defocus_distance_m,
        )

        fzp_pixel_geometry = _fzp_plane_pixel_geometry()
        open_area_m2 = (
            _zone_plate_annulus().sum() * fzp_pixel_geometry.width_m * fzp_pixel_geometry.height_m
        )
        probe_power = intensity(probe.get_array()[0]).sum() * _FZP_PROBE_PITCH_M**2
        assert probe_power == pytest.approx(open_area_m2, rel=1e-9)

    def test_zero_total_distance_raises(self) -> None:
        """``defocus = -f`` collapses the conjugate grid; it must not return NaNs."""
        with pytest.raises(ValueError, match='nonzero'):
            generate_fresnel_zone_plate_probe(
                _fzp_probe_geometry(),
                _FZP_ZONE_PLATE,
                probe_wavelength_m=_FZP_WAVELENGTH_M,
                defocus_distance_m=-_FZP_FOCAL_LENGTH_M,
            )


class TestPropagateProbe:
    """propagate_probe (renamed from defocus_probe) over the angular spectrum."""

    @staticmethod
    def _probe(num_modes: int) -> Probe:
        rng = numpy.random.default_rng(11)
        y, x = numpy.mgrid[0:32, 0:32] - 16
        envelope = numpy.exp(-(x**2 + y**2) / 60.0)
        array = numpy.stack(
            [envelope * numpy.exp(1j * rng.uniform(0.0, 0.2, (32, 32))) for _ in range(num_modes)]
        )
        return Probe(
            array=array.astype(complex),
            pixel_geometry=PixelGeometry(width_m=1e-8, height_m=1e-8),
        )

    def test_zero_distance_is_an_identity(self) -> None:
        probe = self._probe(1)
        propagated = propagate_probe(probe, probe_wavelength_m=1e-10, propagation_distance_m=0.0)

        numpy.testing.assert_allclose(propagated.get_array(), probe.get_array(), atol=1e-12)

    def test_every_mode_propagates_independently(self) -> None:
        """The propagator ffts the last two axes, so the mode axis must ride through
        untouched -- propagating a stack must equal propagating each mode alone."""
        probe = self._probe(3)
        together = propagate_probe(
            probe, probe_wavelength_m=1e-10, propagation_distance_m=5e-5
        ).get_array()

        for index in range(3):
            alone = propagate_probe(
                Probe(
                    array=probe.get_array()[index : index + 1],
                    pixel_geometry=probe.get_pixel_geometry(),
                ),
                probe_wavelength_m=1e-10,
                propagation_distance_m=5e-5,
            ).get_array()
            numpy.testing.assert_allclose(together[index], alone[0], atol=1e-12)

    def test_propagation_is_reversible(self) -> None:
        probe = self._probe(2)
        forward = propagate_probe(probe, probe_wavelength_m=1e-10, propagation_distance_m=5e-5)
        back = propagate_probe(forward, probe_wavelength_m=1e-10, propagation_distance_m=-5e-5)

        numpy.testing.assert_allclose(back.get_array(), probe.get_array(), atol=1e-9)


_KB_NUM_PX = 256
_KB_WAVELENGTH_M = 1.24e-10
_KB_PROBE_PITCH_M = 4e-9
_KB_GRAZING_ANGLE_RAD = 3e-3
_KB_FOCUS_DISTANCE_M = 0.05
_SINC_FWHM_FACTOR = 0.886


def _kb_probe_geometry() -> ProbeGeometry:
    return ProbeGeometry(
        width_px=_KB_NUM_PX,
        height_px=_KB_NUM_PX,
        pixel_width_m=_KB_PROBE_PITCH_M,
        pixel_height_m=_KB_PROBE_PITCH_M,
    )


def _kb_mirrors(
    numerical_aperture_x: float = 1.5e-3, numerical_aperture_y: float = 1.5e-3
) -> KirkpatrickBaezMirrorPair:
    return KirkpatrickBaezMirrorPair(
        horizontal=KirkpatrickBaezMirror.from_numerical_aperture(
            numerical_aperture_x,
            focus_distance_m=_KB_FOCUS_DISTANCE_M,
            grazing_angle_rad=_KB_GRAZING_ANGLE_RAD,
        ),
        vertical=KirkpatrickBaezMirror.from_numerical_aperture(
            numerical_aperture_y,
            focus_distance_m=_KB_FOCUS_DISTANCE_M,
            grazing_angle_rad=_KB_GRAZING_ANGLE_RAD,
        ),
    )


def _kb_pupil_mask(mirrors: KirkpatrickBaezMirrorPair, defocus_distance_m: float) -> numpy.ndarray:
    """Rebuild the binary pupil the generator forms, for an energy-conservation reference."""
    reference_distance_m = mirrors.get_reference_distance_m()
    pupil_pixel_geometry = compute_far_field_pixel_geometry(
        _kb_probe_geometry().get_pixel_geometry(),
        ImageExtent(width_px=_KB_NUM_PX, height_px=_KB_NUM_PX),
        wavelength_m=_KB_WAVELENGTH_M,
        propagation_distance_m=reference_distance_m + defocus_distance_m,
    )
    lx = pupil_pixel_geometry.width_m * (numpy.arange(_KB_NUM_PX) - _KB_NUM_PX // 2)
    ly = pupil_pixel_geometry.height_m * (numpy.arange(_KB_NUM_PX) - _KB_NUM_PX // 2)
    yy, xx = numpy.meshgrid(ly, lx, indexing='ij')
    return numpy.logical_and(
        numpy.fabs(xx) <= mirrors.horizontal.get_numerical_aperture() * reference_distance_m,
        numpy.fabs(yy) <= mirrors.vertical.get_numerical_aperture() * reference_distance_m,
    )


def _interpolated_fwhm(profile: numpy.ndarray, pitch_m: float) -> float:
    """FWHM of a 1D profile with linear interpolation of the two half-maximum crossings.

    A profile still above half maximum at an array edge is reported out to that edge, so a
    beam wider than the window reads as the window rather than raising.
    """
    normalized = profile / profile.max()
    above = numpy.flatnonzero(normalized >= 0.5)
    lower, upper = int(above[0]), int(above[-1])

    if lower > 0:
        left = lower - (0.5 - normalized[lower - 1]) / (normalized[lower] - normalized[lower - 1])
    else:
        left = 0.0

    if upper < len(normalized) - 1:
        right = upper + (normalized[upper] - 0.5) / (normalized[upper] - normalized[upper + 1])
    else:
        right = float(len(normalized) - 1)

    return float((right - left) * pitch_m)


def _kb_central_cuts(probe: Probe) -> tuple[numpy.ndarray, numpy.ndarray]:
    probe_intensity = intensity(probe.get_array()[0])
    return probe_intensity[_KB_NUM_PX // 2, :], probe_intensity[:, _KB_NUM_PX // 2]


def _kb_peak_position_m(profile: numpy.ndarray) -> float:
    """Sub-pixel peak position relative to the array center, by parabolic refinement."""
    peak = int(numpy.argmax(profile))
    before, at, after = profile[peak - 1], profile[peak], profile[peak + 1]
    offset = 0.5 * (before - after) / (before - 2 * at + after)
    return float((peak + offset - (_KB_NUM_PX - 1) / 2) * _KB_PROBE_PITCH_M)


class TestKirkpatrickBaezMirror:
    def test_projected_aperture_is_the_grazing_projection(self) -> None:
        mirror = KirkpatrickBaezMirror(
            acceptance_length_m=0.1, grazing_angle_rad=3e-3, focus_distance_m=0.05
        )

        assert mirror.get_projected_aperture_m() == pytest.approx(0.1 * numpy.sin(3e-3), rel=1e-12)

    def test_numerical_aperture_inverts_the_alternate_constructor(self) -> None:
        """from_numerical_aperture must be the exact inverse of get_numerical_aperture."""
        mirror = KirkpatrickBaezMirror.from_numerical_aperture(
            1.25e-3, focus_distance_m=0.04, grazing_angle_rad=2.5e-3
        )

        assert mirror.get_numerical_aperture() == pytest.approx(1.25e-3, rel=1e-12)
        assert mirror.focus_distance_m == 0.04
        assert mirror.grazing_angle_rad == 2.5e-3

    def test_collimated_input_makes_the_focal_length_the_focus_distance(self) -> None:
        """A zero source distance denotes a collimated beam, where p q / (p + q) degenerates."""
        mirror = KirkpatrickBaezMirror(
            acceptance_length_m=0.1, grazing_angle_rad=3e-3, focus_distance_m=0.05
        )

        assert mirror.get_focal_length_m() == pytest.approx(0.05, rel=1e-12)

    def test_finite_source_distance_uses_the_thin_lens_formula(self) -> None:
        mirror = KirkpatrickBaezMirror(
            acceptance_length_m=0.1,
            grazing_angle_rad=3e-3,
            focus_distance_m=0.05,
            source_distance_m=45.0,
        )

        assert mirror.get_focal_length_m() == pytest.approx(45.0 * 0.05 / 45.05, rel=1e-12)

    @pytest.mark.parametrize(
        'override',
        [
            {'acceptance_length_m': 0.0},
            {'acceptance_length_m': -0.1},
            {'acceptance_length_m': math.inf},
            {'grazing_angle_rad': 0.0},
            {'grazing_angle_rad': 0.5 * math.pi},
            {'grazing_angle_rad': 2.0},
            {'focus_distance_m': 0.0},
            {'focus_distance_m': math.nan},
            {'focus_distance_m': math.inf},
            {'source_distance_m': -1.0},
        ],
    )
    def test_rejects_geometry_that_is_not_a_mirror(self, override: dict) -> None:
        """Rejected at construction, and the error names the field that is wrong.

        A zero focus distance would otherwise surface as a ZeroDivisionError from inside
        the numerical-aperture accessor, several frames from the value that caused it.
        Infinity needs saying separately: a bare positivity constraint admits it, since
        ``inf > 0`` holds.
        """
        kwargs = dict(acceptance_length_m=0.1, grazing_angle_rad=3e-3, focus_distance_m=0.05)
        kwargs.update(override)
        offending_field = next(iter(override))

        with pytest.raises(ValidationError, match=offending_field):
            KirkpatrickBaezMirror(**kwargs)

    def test_rejects_an_unknown_field(self) -> None:
        """A preset is a bare literal call, so a misspelled field must not pass silently."""
        with pytest.raises(ValidationError, match='grazing_angle'):
            # Misspelled on purpose; mypy agreeing the field is unknown is the point.
            KirkpatrickBaezMirror(  # type: ignore[call-arg]
                acceptance_length_m=0.1,
                grazing_angle=3e-3,
                focus_distance_m=0.05,
            )

    def test_a_pair_reports_which_mirror_is_wrong(self) -> None:
        """A nested mirror is validated as part of the pair, and the error locates it."""
        sound = dict(acceptance_length_m=0.1, grazing_angle_rad=3e-3, focus_distance_m=0.05)

        with pytest.raises(ValidationError, match='horizontal'):
            # Pydantic coerces each mapping into a KirkpatrickBaezMirror; that
            # coercion is what the nested-validation message under test comes from.
            KirkpatrickBaezMirrorPair(
                horizontal={**sound, 'focus_distance_m': 0.0},  # type: ignore[arg-type]
                vertical=sound,  # type: ignore[arg-type]
            )

    def test_reference_distance_is_the_mean_of_the_two_mirrors(self) -> None:
        pair = KirkpatrickBaezMirrorPair(
            horizontal=KirkpatrickBaezMirror(
                acceptance_length_m=0.1, grazing_angle_rad=3e-3, focus_distance_m=0.04
            ),
            vertical=KirkpatrickBaezMirror(
                acceptance_length_m=0.1, grazing_angle_rad=3e-3, focus_distance_m=0.06
            ),
        )

        assert pair.get_reference_distance_m() == pytest.approx(0.05, rel=1e-12)


class TestGenerateKbMirrorProbe:
    def test_declares_the_requested_geometry(self) -> None:
        geometry = _kb_probe_geometry()

        probe = generate_kb_mirror_probe(
            geometry, _kb_mirrors(), probe_wavelength_m=_KB_WAVELENGTH_M
        )

        assert probe.get_array().shape == (1, _KB_NUM_PX, _KB_NUM_PX)
        assert probe.get_pixel_geometry() == geometry.get_pixel_geometry()
        assert numpy.all(numpy.isfinite(probe.get_array()))

    @pytest.mark.parametrize(
        ('numerical_aperture_x', 'numerical_aperture_y'),
        [(1.5e-3, 1.5e-3), (1.5e-3, 7.5e-4), (1.0e-3, 2.0e-3)],
    )
    def test_focus_width_matches_the_uniform_slit_prediction(
        self, numerical_aperture_x: float, numerical_aperture_y: float
    ) -> None:
        """A uniformly filled slit of aperture NA focuses to 0.886 lambda / (2 NA) FWHM.

        Each axis takes the numerical aperture of its own mirror, so an anisotropic pair
        gives an anisotropic focus. The tolerance is loose because the FWHM is read off a
        grid whose pitch is a tenth of the focus.
        """
        mirrors = _kb_mirrors(numerical_aperture_x, numerical_aperture_y)

        probe = generate_kb_mirror_probe(
            _kb_probe_geometry(), mirrors, probe_wavelength_m=_KB_WAVELENGTH_M
        )

        cut_x, cut_y = _kb_central_cuts(probe)
        expected_x = _SINC_FWHM_FACTOR * _KB_WAVELENGTH_M / (2.0 * numerical_aperture_x)
        expected_y = _SINC_FWHM_FACTOR * _KB_WAVELENGTH_M / (2.0 * numerical_aperture_y)

        assert _interpolated_fwhm(cut_x, _KB_PROBE_PITCH_M) == pytest.approx(expected_x, rel=0.08)
        assert _interpolated_fwhm(cut_y, _KB_PROBE_PITCH_M) == pytest.approx(expected_y, rel=0.08)

    def test_astigmatism_puts_each_axis_waist_on_its_own_plane(self) -> None:
        """The x waist sits half the astigmatism upstream of the reference plane, y downstream.

        This is the property the single-propagation construction exists to deliver: a
        per-axis focus distance written into the pupil as residual curvature rather than
        as two separate propagations. Scanning defocus and locating each axis's narrowest
        plane tests the placement rather than merely that the two axes differ.
        """
        mirrors = _kb_mirrors()
        astigmatism_m = 200e-6
        defocus_distances_m = numpy.linspace(-1.5, 1.5, 7) * 0.5 * astigmatism_m

        widths_x = []
        widths_y = []

        for defocus_distance_m in defocus_distances_m:
            probe = generate_kb_mirror_probe(
                _kb_probe_geometry(),
                mirrors,
                probe_wavelength_m=_KB_WAVELENGTH_M,
                defocus_distance_m=float(defocus_distance_m),
                astigmatism_m=astigmatism_m,
            )
            cut_x, cut_y = _kb_central_cuts(probe)
            widths_x.append(_interpolated_fwhm(cut_x, _KB_PROBE_PITCH_M))
            widths_y.append(_interpolated_fwhm(cut_y, _KB_PROBE_PITCH_M))

        assert defocus_distances_m[int(numpy.argmin(widths_x))] == pytest.approx(
            -0.5 * astigmatism_m
        )
        assert defocus_distances_m[int(numpy.argmin(widths_y))] == pytest.approx(
            +0.5 * astigmatism_m
        )

        # At its own waist each axis reaches the uniform-slit limit for its aperture.
        expected_m = _SINC_FWHM_FACTOR * _KB_WAVELENGTH_M / (2.0 * 1.5e-3)

        assert min(widths_x) == pytest.approx(expected_m, rel=0.08)
        assert min(widths_y) == pytest.approx(expected_m, rel=0.08)

    def test_probe_is_separable_without_figure_error(self) -> None:
        """A rectangular pupil with per-axis phase is a rank-one outer product."""
        probe = generate_kb_mirror_probe(
            _kb_probe_geometry(),
            _kb_mirrors(1.5e-3, 1.0e-3),
            probe_wavelength_m=_KB_WAVELENGTH_M,
            astigmatism_m=0.004,
        )

        array = probe.get_array()[0]

        assert numpy.linalg.matrix_rank(array, tol=1e-9 * numpy.abs(array).max()) == 1

    @pytest.mark.parametrize('defocus_distance_m', [0.0, -2 * _KB_FOCUS_DISTANCE_M])
    def test_conserves_energy_between_the_pupil_and_probe_planes(
        self, defocus_distance_m: float
    ) -> None:
        """Pins the grid each direction outputs on, independent of phase.

        The backward branch reuses the probe-plane pitch, so a sign slip in the pitch
        selection would show up here as a scale error rather than as a wrong-looking probe.
        """
        mirrors = _kb_mirrors(1.5e-3, 1.0e-3)
        pupil_pixel_geometry = compute_far_field_pixel_geometry(
            _kb_probe_geometry().get_pixel_geometry(),
            ImageExtent(width_px=_KB_NUM_PX, height_px=_KB_NUM_PX),
            wavelength_m=_KB_WAVELENGTH_M,
            propagation_distance_m=mirrors.get_reference_distance_m() + defocus_distance_m,
        )
        open_area_m2 = (
            _kb_pupil_mask(mirrors, defocus_distance_m).sum()
            * pupil_pixel_geometry.width_m
            * pupil_pixel_geometry.height_m
        )

        probe = generate_kb_mirror_probe(
            _kb_probe_geometry(),
            mirrors,
            probe_wavelength_m=_KB_WAVELENGTH_M,
            defocus_distance_m=defocus_distance_m,
        )
        probe_power = intensity(probe.get_array()[0]).sum() * _KB_PROBE_PITCH_M**2

        assert probe_power == pytest.approx(open_area_m2, rel=1e-9)

    def test_rejects_a_vanishing_propagation_distance(self) -> None:
        mirrors = _kb_mirrors()

        with pytest.raises(ValueError, match='nonzero'):
            generate_kb_mirror_probe(
                _kb_probe_geometry(),
                mirrors,
                probe_wavelength_m=_KB_WAVELENGTH_M,
                defocus_distance_m=-mirrors.get_reference_distance_m(),
            )

    def test_rejects_an_aperture_wider_than_the_pupil_window(self) -> None:
        """A coarse probe pitch shrinks the conjugate window until the NA no longer fits.

        Clipping it silently would quietly reduce the numerical aperture and so widen the
        focus, which is indistinguishable from having asked for a different optic.
        """
        coarse_geometry = ProbeGeometry(
            width_px=_KB_NUM_PX,
            height_px=_KB_NUM_PX,
            pixel_width_m=1e-7,
            pixel_height_m=1e-7,
        )

        with pytest.raises(ValueError, match='does not fit the pupil window'):
            generate_kb_mirror_probe(
                coarse_geometry, _kb_mirrors(), probe_wavelength_m=_KB_WAVELENGTH_M
            )

    def test_rejects_astigmatism_that_places_a_focus_at_the_pupil(self) -> None:
        mirrors = _kb_mirrors()

        with pytest.raises(ValueError, match='Astigmatism'):
            generate_kb_mirror_probe(
                _kb_probe_geometry(),
                mirrors,
                probe_wavelength_m=_KB_WAVELENGTH_M,
                astigmatism_m=2 * mirrors.get_reference_distance_m(),
            )

    def test_gaussian_truncation_suppresses_the_sinc_side_lobes(self) -> None:
        """Tapering the pupil trades focus width for side-lobe power, monotonically.

        A hard-edged pupil produces the sinc side lobes that a real KB focus does not
        have; this is the knob that removes them.
        """
        mirrors = _kb_mirrors()
        aperture_m = (
            2.0 * mirrors.horizontal.get_numerical_aperture() * (mirrors.get_reference_distance_m())
        )

        def first_side_lobe_ratio(incident_beam_fwhm_m: float) -> float:
            probe = generate_kb_mirror_probe(
                _kb_probe_geometry(),
                mirrors,
                probe_wavelength_m=_KB_WAVELENGTH_M,
                incident_beam_fwhm_x_m=incident_beam_fwhm_m,
            )
            cut = _kb_central_cuts(probe)[0]
            # Tapering widens the main lobe, so the wing has to start at the first
            # minimum past the peak rather than at a fixed pixel offset.
            half = cut[int(numpy.argmax(cut)) :]
            descending = numpy.flatnonzero(numpy.diff(half) >= 0.0)
            return float(half[int(descending[0]) + 1 :].max() / cut.max())

        uniform = first_side_lobe_ratio(0.0)
        tapered = first_side_lobe_ratio(0.6 * aperture_m)
        strongly_tapered = first_side_lobe_ratio(0.3 * aperture_m)

        assert tapered < uniform
        assert strongly_tapered < tapered

    def test_a_very_wide_incident_beam_recovers_the_uniform_pupil(self) -> None:
        mirrors = _kb_mirrors()
        aperture_m = (
            2.0 * mirrors.horizontal.get_numerical_aperture() * (mirrors.get_reference_distance_m())
        )

        uniform = generate_kb_mirror_probe(
            _kb_probe_geometry(), mirrors, probe_wavelength_m=_KB_WAVELENGTH_M
        )
        nearly_uniform = generate_kb_mirror_probe(
            _kb_probe_geometry(),
            mirrors,
            probe_wavelength_m=_KB_WAVELENGTH_M,
            incident_beam_fwhm_x_m=1e4 * aperture_m,
            incident_beam_fwhm_y_m=1e4 * aperture_m,
        )

        numpy.testing.assert_allclose(
            nearly_uniform.get_array(), uniform.get_array(), rtol=1e-6, atol=1e-12
        )

    def test_piston_figure_error_is_a_global_phase(self) -> None:
        """Legendre order zero adds a constant height, which cannot change the intensity."""
        geometry = _kb_probe_geometry()
        mirrors = _kb_mirrors()

        plain = generate_kb_mirror_probe(
            geometry, mirrors, probe_wavelength_m=_KB_WAVELENGTH_M
        ).get_array()
        with_piston = generate_kb_mirror_probe(
            geometry,
            mirrors,
            probe_wavelength_m=_KB_WAVELENGTH_M,
            figure_error_x=[LegendreMode(coefficient_m=3e-9, order=0)],
        ).get_array()

        numpy.testing.assert_allclose(intensity(with_piston), intensity(plain), rtol=1e-9)

    def test_second_order_figure_error_is_equivalent_to_a_defocus(self) -> None:
        """P_2 is a quadratic height error, so it adds pupil curvature like a defocus does.

        Matching the quadratic coefficients gives 1/z_eff = 1/z_ref + 6 sin(theta) c2 / a^2,
        with a the pupil half-aperture. The probe carrying the figure error at zero defocus
        must then be as sharp as a clean probe explicitly defocused to that plane.
        """
        geometry = _kb_probe_geometry()
        mirrors = _kb_mirrors()
        reference_distance_m = mirrors.get_reference_distance_m()
        half_aperture_m = mirrors.horizontal.get_numerical_aperture() * reference_distance_m
        coefficient_m = 2.0e-9

        effective_distance_m = 1.0 / (
            1.0 / reference_distance_m
            + 6.0 * numpy.sin(_KB_GRAZING_ANGLE_RAD) * coefficient_m / half_aperture_m**2
        )
        equivalent_defocus_m = effective_distance_m - reference_distance_m

        def sharpness(probe: Probe) -> float:
            cut = _kb_central_cuts(probe)[0]
            return float(numpy.square(cut).sum() / numpy.square(cut.sum()))

        from_figure_error = sharpness(
            generate_kb_mirror_probe(
                geometry,
                mirrors,
                probe_wavelength_m=_KB_WAVELENGTH_M,
                figure_error_x=[LegendreMode(coefficient_m=coefficient_m, order=2)],
            )
        )
        from_defocus = sharpness(
            generate_kb_mirror_probe(
                geometry,
                mirrors,
                probe_wavelength_m=_KB_WAVELENGTH_M,
                defocus_distance_m=equivalent_defocus_m,
            )
        )

        assert equivalent_defocus_m < 0.0
        assert from_figure_error == pytest.approx(from_defocus, rel=1e-3)

    def test_first_order_figure_error_shifts_the_focus_laterally(self) -> None:
        """P_1 is a tilt, which steers the focus by 2 z sin(theta) c1 / a.

        The comparison is differential because the probe grid is centered on N // 2 while
        the transverse-coordinate convention centers on (N - 1) / 2; that half-pixel offset
        is common to every propagator output here and cancels between the two signs.
        """
        geometry = _kb_probe_geometry()
        mirrors = _kb_mirrors()
        reference_distance_m = mirrors.get_reference_distance_m()
        half_aperture_m = mirrors.horizontal.get_numerical_aperture() * reference_distance_m
        coefficient_m = 1.0e-8

        def peak_position_m(signed_coefficient_m: float) -> float:
            probe = generate_kb_mirror_probe(
                geometry,
                mirrors,
                probe_wavelength_m=_KB_WAVELENGTH_M,
                figure_error_x=[LegendreMode(coefficient_m=signed_coefficient_m, order=1)],
            )
            return _kb_peak_position_m(_kb_central_cuts(probe)[0])

        measured = peak_position_m(coefficient_m) - peak_position_m(-coefficient_m)
        expected = (
            -4.0
            * reference_distance_m
            * numpy.sin(_KB_GRAZING_ANGLE_RAD)
            * coefficient_m
            / half_aperture_m
        )

        assert measured == pytest.approx(expected, rel=0.02)


class TestGenerateMirrorFigureError:
    def test_realized_rms_slope_matches_the_request(self) -> None:
        """Legendre derivatives are not orthogonal, so the whole series is rescaled at once.

        Scaling each coefficient by 1 / sqrt(n (n + 1)) would be wrong: the Gram matrix of
        the derivatives has off-diagonal entries as large as its diagonal.
        """
        acceptance_length_m = 0.1
        rms_slope_error_rad = 100e-9

        for seed in range(4):
            modes = generate_mirror_figure_error(
                numpy.random.default_rng(seed),
                rms_slope_error_rad=rms_slope_error_rad,
                acceptance_length_m=acceptance_length_m,
                num_modes=10,
            )

            assert _realized_rms_slope(modes, acceptance_length_m) == pytest.approx(
                rms_slope_error_rad, rel=1e-9
            )

    def test_is_reproducible_from_the_seed(self) -> None:
        kwargs: dict[str, Any] = dict(
            rms_slope_error_rad=50e-9, acceptance_length_m=0.1, num_modes=8
        )

        first = generate_mirror_figure_error(numpy.random.default_rng(7), **kwargs)
        again = generate_mirror_figure_error(numpy.random.default_rng(7), **kwargs)
        other = generate_mirror_figure_error(numpy.random.default_rng(8), **kwargs)

        assert first == again
        assert first != other

    def test_omits_orders_below_the_lowest_requested(self) -> None:
        """The default skips piston and tilt, as mirror metrology quotes residual figure."""
        modes = generate_mirror_figure_error(
            numpy.random.default_rng(0),
            rms_slope_error_rad=50e-9,
            acceptance_length_m=0.1,
            num_modes=6,
        )

        assert [mode.order for mode in modes] == [2, 3, 4, 5, 6, 7]

    def test_more_negative_psd_exponent_concentrates_power_at_low_orders(self) -> None:
        def coefficient_decay(psd_exponent: float) -> float:
            modes = generate_mirror_figure_error(
                numpy.random.default_rng(3),
                rms_slope_error_rad=50e-9,
                acceptance_length_m=0.1,
                num_modes=8,
                psd_exponent=psd_exponent,
            )
            magnitudes = numpy.fabs([mode.coefficient_m for mode in modes])
            return float(magnitudes[0] / magnitudes[-1])

        assert coefficient_decay(-4.0) > coefficient_decay(-2.0) > coefficient_decay(-1.0)

    def test_a_vanishing_slope_error_gives_a_flat_mirror(self) -> None:
        modes = generate_mirror_figure_error(
            numpy.random.default_rng(0),
            rms_slope_error_rad=0.0,
            acceptance_length_m=0.1,
            num_modes=5,
        )

        assert all(mode.coefficient_m == 0.0 for mode in modes)

    @pytest.mark.parametrize(
        ('override', 'match'),
        [
            ({'num_modes': 0}, 'at least one'),
            ({'lowest_order': -1}, 'non-negative'),
            ({'rms_slope_error_rad': -1.0}, 'non-negative'),
            ({'acceptance_length_m': 0.0}, 'positive'),
        ],
    )
    def test_rejects_invalid_arguments(self, override: dict, match: str) -> None:
        kwargs: dict[str, Any] = dict(
            rms_slope_error_rad=50e-9, acceptance_length_m=0.1, num_modes=5
        )
        kwargs.update(override)

        with pytest.raises(ValueError, match=match):
            generate_mirror_figure_error(numpy.random.default_rng(0), **kwargs)


def _realized_rms_slope(modes: list[LegendreMode], acceptance_length_m: float) -> float:
    """Independent quadrature over the aperture, with more nodes than the generator uses."""
    highest_order = max(mode.order for mode in modes)
    series = numpy.zeros(highest_order + 1)

    for mode in modes:
        series[mode.order] = mode.coefficient_m

    derivative = numpy.polynomial.legendre.legder(series)
    nodes, weights = numpy.polynomial.legendre.leggauss(4 * (highest_order + 1))
    slope = 2.0 * numpy.polynomial.legendre.legval(nodes, derivative) / acceptance_length_m
    return float(numpy.sqrt(0.5 * numpy.sum(weights * numpy.square(slope))))


def _gaussian_probe(sigma_m: float = 1.2e-7, num_px: int = 64) -> Probe:
    pixel_geometry = PixelGeometry(width_m=1e-8, height_m=1e-8)
    coordinate = (numpy.arange(num_px) - (num_px - 1) / 2) * pixel_geometry.width_m
    y, x = numpy.meshgrid(coordinate, coordinate, indexing='ij')
    array = numpy.exp(-(numpy.square(x) + numpy.square(y)) / (4 * sigma_m**2)).astype(complex)
    return Probe(array=array, pixel_geometry=pixel_geometry)


def _gsm(coherence_length_y_m: float = 1.2e-7) -> GaussianSchellStrategy:
    return GaussianSchellStrategy(
        beam_size_x_m=1.2e-7,
        coherence_length_x_m=1.2e-7,
        beam_size_y_m=1.2e-7,
        coherence_length_y_m=coherence_length_y_m,
    )


class TestGaussianSchellStrategy:
    def test_weight_ratio_matches_the_closed_form_at_equal_sigma_and_xi(self) -> None:
        """With sigma == xi the geometric ratio b / (a + b + c) collapses to (3 - sqrt(5)) / 2.

        a = 1 / (4 sigma^2) and b = 1 / (2 sigma^2) = 2 a give c = a sqrt(5), so the ratio
        is 2 / (3 + sqrt(5)) exactly, independent of the beam size itself.
        """
        weights = _gsm(coherence_length_y_m=1e9).get_imode_weights(6)
        expected_ratio = (3.0 - numpy.sqrt(5.0)) / 2.0

        assert weights[1] / weights[0] == pytest.approx(expected_ratio, rel=1e-12)

    def test_the_coherent_limit_leaves_a_single_occupied_mode(self) -> None:
        strategy = GaussianSchellStrategy(
            beam_size_x_m=1.2e-7,
            coherence_length_x_m=1e9,
            beam_size_y_m=1.2e-7,
            coherence_length_y_m=1e9,
        )
        weights = strategy.get_imode_weights(4)

        assert weights[0] / weights.sum() == pytest.approx(1.0, rel=1e-12)

    def test_anisotropic_coherence_puts_the_extra_modes_on_the_incoherent_axis(self) -> None:
        """A KB beam is typically far less coherent in one direction than the other."""
        strategy = GaussianSchellStrategy(
            beam_size_x_m=1.2e-7,
            coherence_length_x_m=6e-8,
            beam_size_y_m=1.2e-7,
            coherence_length_y_m=1e9,
        )

        assert strategy.get_imode_orders(5) == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    def test_weights_decrease_with_mode_index(self) -> None:
        weights = _gsm(coherence_length_y_m=2.4e-7).get_imode_weights(6)

        assert numpy.all(numpy.diff(weights) <= 0.0)

    def test_gram_schmidt_keeps_the_dominant_mode_and_orthogonalizes_the_rest(self) -> None:
        """Mode zero must survive as the input probe so the analytic weights stay attached.

        An SVD basis for the same span would not: it mixes every mode into every output,
        after which the eigenvalue spectrum describes nothing in particular.
        """
        probe = _gaussian_probe()
        num_imodes = 5

        result = generate_incoherent_probe_modes(probe, num_imodes, strategy=_gsm())
        rows = result.get_array().reshape(num_imodes, -1)
        unit_rows = rows / numpy.linalg.norm(rows, axis=1, keepdims=True)
        dominant = probe.get_array()[0].ravel()

        overlap = numpy.abs(unit_rows @ unit_rows.conj().T)

        assert numpy.abs(overlap - numpy.eye(num_imodes)).max() < 1e-10
        assert abs(numpy.vdot(unit_rows[0], dominant / numpy.linalg.norm(dominant))) == (
            pytest.approx(1.0, rel=1e-12)
        )

    def test_realized_power_follows_the_analytic_spectrum(self) -> None:
        """A single-mode input has nothing to preserve, so the predicted spectrum applies."""
        probe = _gaussian_probe()
        strategy = _gsm(coherence_length_y_m=2.4e-7)

        result = generate_incoherent_probe_modes(probe, 5, strategy=strategy)
        array = result.get_array()
        realized = numpy.array([intensity(mode).sum() for mode in array])
        expected = strategy.get_imode_weights(5)

        numpy.testing.assert_allclose(
            realized / realized.sum(), expected / expected.sum(), rtol=1e-9
        )
        assert intensity(array).sum() == pytest.approx(intensity(probe.get_array()).sum(), rel=1e-9)

    @pytest.mark.parametrize(
        'override',
        [
            {'beam_size_x_m': 0.0},
            {'coherence_length_x_m': -1.0},
            {'beam_size_y_m': 0.0},
            {'coherence_length_y_m': 0.0},
        ],
    )
    def test_rejects_invalid_arguments(self, override: dict) -> None:
        kwargs = dict(
            beam_size_x_m=1.2e-7,
            coherence_length_x_m=1.2e-7,
            beam_size_y_m=1.2e-7,
            coherence_length_y_m=1.2e-7,
        )
        kwargs.update(override)

        with pytest.raises(ValueError):
            GaussianSchellStrategy(**kwargs)


class TestKbMirrorProbeSamplingWarnings:
    """The two sampling warnings pull in opposite directions and both scale with the grid.

    A finer probe pitch shrinks the conjugate window and so the pupil footprint, while a
    coarser one undersamples the focus; a grid too small satisfies neither.
    """

    def test_warns_when_the_pupil_is_too_coarse_for_figure_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A fine probe pitch shrinks the conjugate window, so the aperture lands on few pixels.
        fine_geometry = ProbeGeometry(
            width_px=_KB_NUM_PX,
            height_px=_KB_NUM_PX,
            pixel_width_m=2e-10,
            pixel_height_m=2e-10,
        )

        with caplog.at_level('WARNING', logger='ptychodus.api.simulate.probe'):
            generate_kb_mirror_probe(
                fine_geometry,
                _kb_mirrors(),
                probe_wavelength_m=_KB_WAVELENGTH_M,
                figure_error_x=[LegendreMode(coefficient_m=1e-9, order=3)],
            )

        assert 'figure error' in caplog.text

    def test_stays_quiet_about_the_pupil_without_figure_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A coarsely sampled pupil only matters once something is written onto it."""
        fine_geometry = ProbeGeometry(
            width_px=_KB_NUM_PX,
            height_px=_KB_NUM_PX,
            pixel_width_m=2e-10,
            pixel_height_m=2e-10,
        )

        with caplog.at_level('WARNING', logger='ptychodus.api.simulate.probe'):
            generate_kb_mirror_probe(
                fine_geometry, _kb_mirrors(), probe_wavelength_m=_KB_WAVELENGTH_M
            )

        assert 'figure error' not in caplog.text

    def test_warns_when_the_focus_is_undersampled(self, caplog: pytest.LogCaptureFixture) -> None:
        """A high numerical aperture on a coarse grid puts the focus below two pixels."""
        with caplog.at_level('WARNING', logger='ptychodus.api.simulate.probe'):
            generate_kb_mirror_probe(
                _kb_probe_geometry(),
                _kb_mirrors(1.0e-2, 1.0e-2),
                probe_wavelength_m=_KB_WAVELENGTH_M,
            )

        assert 'undersampled' in caplog.text

    def test_stays_quiet_at_the_default_sampling(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level('WARNING', logger='ptychodus.api.simulate.probe'):
            generate_kb_mirror_probe(
                _kb_probe_geometry(),
                _kb_mirrors(),
                probe_wavelength_m=_KB_WAVELENGTH_M,
                figure_error_x=[LegendreMode(coefficient_m=1e-9, order=3)],
            )

        assert caplog.text == ''

    def test_pupil_threshold_is_caller_adjustable(self, caplog: pytest.LogCaptureFixture) -> None:
        """The default sampling is quiet, so demanding a better-resolved pupil must warn."""
        with caplog.at_level('WARNING', logger='ptychodus.api.simulate.probe'):
            generate_kb_mirror_probe(
                _kb_probe_geometry(),
                _kb_mirrors(),
                probe_wavelength_m=_KB_WAVELENGTH_M,
                figure_error_x=[LegendreMode(coefficient_m=1e-9, order=3)],
                min_pupil_px_for_figure_error=64.0,
            )

        assert 'figure error' in caplog.text

    def test_focus_threshold_is_caller_adjustable(self, caplog: pytest.LogCaptureFixture) -> None:
        """Likewise for demanding more probe pixels across the focus."""
        with caplog.at_level('WARNING', logger='ptychodus.api.simulate.probe'):
            generate_kb_mirror_probe(
                _kb_probe_geometry(),
                _kb_mirrors(),
                probe_wavelength_m=_KB_WAVELENGTH_M,
                min_px_per_focus_fwhm=16.0,
            )

        assert 'undersampled' in caplog.text

    def test_thresholds_can_be_silenced(self, caplog: pytest.LogCaptureFixture) -> None:
        """A caller who knows the grid is coarse on purpose can turn both warnings off."""
        with caplog.at_level('WARNING', logger='ptychodus.api.simulate.probe'):
            generate_kb_mirror_probe(
                _kb_probe_geometry(),
                _kb_mirrors(1.0e-2, 1.0e-2),
                probe_wavelength_m=_KB_WAVELENGTH_M,
                figure_error_x=[LegendreMode(coefficient_m=1e-9, order=3)],
                min_pupil_px_for_figure_error=0.0,
                min_px_per_focus_fwhm=0.0,
            )

        assert caplog.text == ''


def _warm_probe(num_imodes: int = 2, power_split: float = 0.1) -> Probe:
    """A converged-looking probe whose higher mode is not polynomial-representable."""
    pixel_geometry = PixelGeometry(width_m=1e-8, height_m=1e-8)
    num_px = 48
    coordinate = (numpy.arange(num_px) - (num_px - 1) / 2) * pixel_geometry.width_m
    y, x = numpy.meshgrid(coordinate, coordinate, indexing='ij')
    radius = numpy.hypot(x, y)
    angle = numpy.arctan2(y, x)
    dominant = numpy.exp(-(x**2 + y**2) / (2 * (1.2e-7) ** 2)).astype(complex)

    modes = [dominant]

    for order in range(1, num_imodes):
        extra = dominant * numpy.exp(1j * order * angle) * numpy.sin(6 * order * radius / 1e-7)
        extra *= numpy.sqrt(power_split * intensity(dominant).sum() / intensity(extra).sum())
        modes.append(extra)

    return Probe(array=numpy.stack(modes), pixel_geometry=pixel_geometry)


def _all_strategies() -> list[IncoherentModeStrategy]:
    return [
        RandomPhaseRampStrategy(numpy.random.default_rng(0)),
        GaussianSchellStrategy(
            beam_size_x_m=1.2e-7,
            coherence_length_x_m=1.2e-7,
            beam_size_y_m=1.2e-7,
            coherence_length_y_m=1.2e-7,
        ),
        ProbeMomentPolynomialStrategy(),
    ]


class TestIncoherentModeContract:
    """Mode handling is shared by every strategy, so every strategy must honor it."""

    @pytest.mark.parametrize('strategy', _all_strategies())
    def test_existing_modes_are_preserved_by_identity(
        self, strategy: IncoherentModeStrategy
    ) -> None:
        """Expanding a converged probe must keep each mode itself, not merely its span.

        An SVD basis preserves the subspace while mixing every input mode into every
        output; measured that way a converged second mode came back with an overlap of
        0.025 against itself. Preserving the span is not preserving the mixed state.
        """
        probe = _warm_probe()
        before = probe.get_array()

        result = generate_incoherent_probe_modes(probe, 4, strategy=strategy)

        rows = result.get_array().reshape(4, -1)
        unit_rows = rows / numpy.linalg.norm(rows, axis=1, keepdims=True)

        for imode in range(before.shape[0]):
            reference = before[imode].ravel()
            reference = reference / numpy.linalg.norm(reference)
            overlap = abs(numpy.vdot(unit_rows[imode], reference))
            assert overlap == pytest.approx(1.0, abs=1e-9), f'mode {imode} overlap {overlap}'

    @pytest.mark.parametrize('strategy', _all_strategies())
    def test_preserved_modes_keep_their_measured_power_ratio(
        self, strategy: IncoherentModeStrategy
    ) -> None:
        """A 90/10 warm start must not be flattened onto the decay profile."""
        probe = _warm_probe()
        before = numpy.array([intensity(mode).sum() for mode in probe.get_array()])
        before /= before.sum()

        array = generate_incoherent_probe_modes(probe, 4, strategy=strategy).get_array()
        after = numpy.array([intensity(mode).sum() for mode in array])

        preserved = after[:2] / after[:2].sum()
        numpy.testing.assert_allclose(preserved, before, rtol=1e-9)

    @pytest.mark.parametrize('strategy', _all_strategies())
    @pytest.mark.parametrize('num_imodes', [1, 2, 3, 5])
    def test_total_power_is_conserved_in_every_direction(
        self, strategy: IncoherentModeStrategy, num_imodes: int
    ) -> None:
        """Expanding, truncating and leaving the count alone must all keep the photons.

        A probe read from file is never rescaled afterward, so whatever this function
        does to the total is final, and the data constrains the probe-object product.
        """
        probe = _warm_probe(num_imodes=3)
        before = intensity(probe.get_array()).sum()

        array = generate_incoherent_probe_modes(probe, num_imodes, strategy=strategy).get_array()

        assert array.shape[0] == num_imodes
        assert intensity(array).sum() == pytest.approx(before, rel=1e-9)

    def test_truncation_keeps_the_strongest_modes(self) -> None:
        probe = _warm_probe(num_imodes=3)
        powers = numpy.array([intensity(mode).sum() for mode in probe.get_array()])
        strongest = probe.get_array()[int(numpy.argmax(powers))].ravel()
        strongest = strongest / numpy.linalg.norm(strongest)

        array = generate_incoherent_probe_modes(probe, 1).get_array()
        kept = array[0].ravel() / numpy.linalg.norm(array[0])

        assert abs(numpy.vdot(kept, strongest)) == pytest.approx(1.0, abs=1e-9)

    def test_matching_the_existing_count_leaves_the_mode_set_alone(self) -> None:
        probe = _warm_probe(num_imodes=3)

        array = generate_incoherent_probe_modes(probe, 3, orthogonalize=False).get_array()

        numpy.testing.assert_allclose(array, probe.get_array(), rtol=1e-9)

    def test_a_dropped_dependent_mode_does_not_cost_photons(self) -> None:
        """A floor that rejects every fill mode must redistribute, not discard, its share."""
        probe = _make_single_mode_probe()
        before = intensity(probe.get_array()).sum()

        array = generate_incoherent_probe_modes(probe, 4, mode_dependence_floor=1.0).get_array()

        assert intensity(array).sum() == pytest.approx(before, rel=1e-9)

    def test_the_default_strategy_is_used_when_none_is_named(self) -> None:
        probe = _make_single_mode_probe()

        implicit = generate_incoherent_probe_modes(probe, 3).get_array()
        explicit = generate_incoherent_probe_modes(
            probe, 3, strategy=DEFAULT_INCOHERENT_MODE_STRATEGY
        ).get_array()

        numpy.testing.assert_array_equal(implicit, explicit)

    def test_rejects_a_non_positive_mode_count(self) -> None:
        with pytest.raises(ValueError, match='at least one'):
            generate_incoherent_probe_modes(_make_single_mode_probe(), 0)


class TestProbeMomentPolynomialStrategy:
    def test_mode_zero_is_the_probe(self) -> None:
        probe = _make_single_mode_probe()
        array = generate_incoherent_probe_modes(probe, 4).get_array()

        dominant = probe.get_array()[0].ravel()
        dominant = dominant / numpy.linalg.norm(dominant)
        mode_zero = array[0].ravel() / numpy.linalg.norm(array[0])

        assert abs(numpy.vdot(mode_zero, dominant)) == pytest.approx(1.0, abs=1e-12)

    def test_decay_settings_set_the_weights(self) -> None:
        strategy = ProbeMomentPolynomialStrategy(
            decay_type=ProbeModeDecayType.EXPONENTIAL, decay_ratio=0.5
        )
        numpy.testing.assert_allclose(
            strategy.get_imode_weights(4), [1.0, 0.5, 0.25, 0.125], rtol=1e-12
        )

    def test_no_decay_leaves_the_later_modes_unoccupied(self) -> None:
        strategy = ProbeMomentPolynomialStrategy(decay_type=ProbeModeDecayType.NONE)
        probe = _make_single_mode_probe()

        array = generate_incoherent_probe_modes(probe, 3, strategy=strategy).get_array()
        powers = numpy.array([intensity(mode).sum() for mode in array])

        assert powers[0] == pytest.approx(intensity(probe.get_array()).sum(), rel=1e-9)
        numpy.testing.assert_allclose(powers[1:], 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        'override', [{'decay_ratio': 0.0}, {'decay_ratio': -1.0}, {'damping_width': 0.0}]
    )
    def test_rejects_invalid_arguments(self, override: dict) -> None:
        with pytest.raises(ValueError):
            ProbeMomentPolynomialStrategy(**override)

    def test_rejects_a_probe_with_no_transverse_extent(self) -> None:
        pixel_geometry = PixelGeometry(width_m=1e-8, height_m=1e-8)
        spike = numpy.zeros((1, 16, 16), dtype=complex)
        spike[0, 8, 8] = 1.0

        with pytest.raises(ValueError, match='no transverse extent'):
            generate_incoherent_probe_modes(Probe(array=spike, pixel_geometry=pixel_geometry), 3)


class TestGramSchmidt:
    @pytest.mark.parametrize('scale', [1e-170, 1e-160, 1.0, 1e160, 1e170])
    def test_survives_rows_far_from_unit_scale(self, scale: float) -> None:
        """numpy.linalg.norm sums squares unguarded, so an unscaled row set silently fails.

        Below about 1e-170 and above about 1e160 both norms collapse to zero or infinity
        and every row is misread as linearly dependent; the degradation starts earlier
        still, reaching 8.85e-06 orthogonality error at 1e-160.
        """
        rng = numpy.random.default_rng(0)
        # astype: numpy's stubs lose complexity through `complex_array * scalar`.
        rows = ((rng.normal(size=(5, 200)) + 1j * rng.normal(size=(5, 200))) * scale).astype(
            complex
        )

        result = _gram_schmidt(rows, dependence_floor=1e-8)
        norms = numpy.linalg.norm(result, axis=1)

        assert numpy.count_nonzero(norms) == 5
        overlap = numpy.abs(result @ result.conj().T)
        assert numpy.abs(overlap - numpy.eye(5)).max() < 1e-12

    def test_orthogonalizes_single_precision_input_at_double_precision(self) -> None:
        rng = numpy.random.default_rng(0)
        rows = (rng.normal(size=(5, 200)) + 1j * rng.normal(size=(5, 200))).astype(numpy.complex64)

        result = _gram_schmidt(rows, dependence_floor=1e-8)
        overlap = numpy.abs(result @ result.conj().T)

        assert result.dtype == numpy.complex128
        assert numpy.abs(overlap - numpy.eye(5)).max() < 1e-12

    def test_drops_a_dependent_row_rather_than_normalizing_roundoff(self) -> None:
        rng = numpy.random.default_rng(0)
        rows = rng.normal(size=(4, 200)) + 1j * rng.normal(size=(4, 200))
        rows[3] = rows[0] + 2 * rows[1]

        result = _gram_schmidt(rows, dependence_floor=1e-8)

        assert numpy.linalg.norm(result[3]) == 0.0
        assert numpy.count_nonzero(numpy.linalg.norm(result, axis=1)) == 3
