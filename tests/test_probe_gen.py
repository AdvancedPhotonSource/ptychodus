"""Unit tests for probe generation functions in ptychodus.api.simulate.probe."""

import numpy
import numpy.testing
import pytest

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import HermiteMode, ImageExtent, PixelGeometry
from ptychodus.api.probe import Probe, ProbeGeometry
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    generate_average_pattern_probe,
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_hermite_probe,
    generate_incoherent_probe_modes,
)
from ptychodus.api.propagate import (
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
        weights = [1.0, 0.5, 0.25, 0.1]

        result = generate_incoherent_probe_modes(rng, probe, weights, orthogonalize=True)

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
        weights = [1.0, 0.5, 0.25]

        result = generate_incoherent_probe_modes(rng, probe, weights, orthogonalize=False)

        # Just verify shape and no NaNs — orthogonality is NOT required here.
        array = result.get_array()
        assert array.shape[0] == len(weights)
        assert not numpy.isnan(array).any()

    def test_single_output_mode_unchanged(self) -> None:
        """A single-element weight list should return a probe with one mode."""
        rng = numpy.random.default_rng(7)
        probe = _make_single_mode_probe()

        result = generate_incoherent_probe_modes(rng, probe, [1.0], orthogonalize=True)

        assert result.get_array().shape[0] == 1

    def test_intensity_weights_respected(self) -> None:
        """Output mode intensities should be proportional to the requested weights."""
        rng = numpy.random.default_rng(99)
        probe = _make_single_mode_probe()
        weights = [4.0, 2.0, 1.0]

        result = generate_incoherent_probe_modes(rng, probe, weights, orthogonalize=True)

        array = result.get_array()
        intensities = numpy.array([numpy.sum(numpy.abs(array[m]) ** 2) for m in range(3)])
        ratios = intensities / intensities[0]
        expected = numpy.array(weights) / weights[0]
        numpy.testing.assert_allclose(ratios, expected, rtol=1e-6)


class TestGenerateCoherentProbeModes:
    def _make_multimode_probe(self, num_imodes: int = 5, seed: int = 3) -> Probe:
        rng = numpy.random.default_rng(seed)
        single = _make_single_mode_probe(seed=seed)
        return generate_incoherent_probe_modes(rng, single, [1.0] * num_imodes, orthogonalize=True)

    def test_eigenmode_incoherent_slots_are_all_nonzero(self) -> None:
        """Regression test: eigenmodes must fill every incoherent mode.

        Before the fix only incoherent slot 0 of each eigenmode was populated,
        leaving zero-power slots that trigger a 0/0 -> NaN in pty-chi's
        Gram-Schmidt orthogonalization (segfault/abort on the CUDA backend).
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
