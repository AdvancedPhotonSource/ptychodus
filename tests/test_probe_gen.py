"""Unit tests for probe generation functions in ptychodus.api.probe_gen."""

import numpy
import numpy.testing

from ptychodus.api.geometry import HermiteMode, PixelGeometry
from ptychodus.api.probe import Probe, ProbeGeometry
from ptychodus.api.probe_gen import generate_hermite_probe, generate_incoherent_probe_modes


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
