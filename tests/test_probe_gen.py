"""Unit tests for probe generation functions in ptychodus.api.probe_gen."""

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import Probe
from ptychodus.api.probe_gen import generate_incoherent_probe_modes


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
