"""Unit tests for ptychodus.api.diffraction.inpaint_bad_pixels and compute_diffraction_summary."""

from collections.abc import Sequence

import numpy
import pytest

from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    DiffractionSummary,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
    compute_diffraction_summary,
    inpaint_bad_pixels,
)
from ptychodus.api.geometry import ImageExtent


def _make_dataset(
    arrays: Sequence[DiffractionArray],
    frame_shape: tuple[int, int],
    bad_pixels: BadPixels | None = None,
) -> SimpleDiffractionDataset:
    """Assemble a SimpleDiffractionDataset from hand-built arrays."""
    height, width = frame_shape
    metadata = DiffractionMetadata(
        num_patterns_per_array=[a.get_num_patterns() for a in arrays],
        pattern_dtype=arrays[0].get_patterns().dtype if arrays else numpy.dtype(numpy.uint16),
        detector_extent=ImageExtent(width_px=width, height_px=height),
    )
    contents_tree = DiffractionDatasetLayoutNode.create_root()
    return SimpleDiffractionDataset(metadata, contents_tree, arrays, bad_pixels)


# ------------------------------ inpaint_bad_pixels ------------------------------


def test_inpaint_returns_input_unchanged_when_mask_is_none() -> None:
    """A None mask short-circuits: the input is returned as-is."""
    pattern = numpy.arange(20, dtype=numpy.float64).reshape(4, 5)
    result = inpaint_bad_pixels(pattern, None)
    assert result is pattern


def test_inpaint_returns_input_unchanged_when_mask_all_false() -> None:
    """An all-good mask short-circuits: the input is returned as-is."""
    pattern = numpy.arange(20, dtype=numpy.float64).reshape(4, 5)
    bad = numpy.zeros((4, 5), dtype=numpy.bool_)
    result = inpaint_bad_pixels(pattern, bad)
    assert result is pattern


def test_inpaint_replaces_bad_positions_and_preserves_good() -> None:
    """Bad-pixel positions are filled by biharmonic; good positions are unchanged."""
    pattern = numpy.ones((8, 8), dtype=numpy.float64) * 5.0
    pattern[3, 3] = 1.0e9
    pattern[4, 4] = -1.0e9
    bad = numpy.zeros((8, 8), dtype=numpy.bool_)
    bad[3, 3] = True
    bad[4, 4] = True

    result = inpaint_bad_pixels(pattern, bad)

    good_mask = numpy.logical_not(bad)
    numpy.testing.assert_array_equal(result[good_mask], pattern[good_mask])
    # Filled values are pulled toward the surrounding constant field of 5.0,
    # not left as the extreme sentinels.
    assert result[3, 3] != pytest.approx(1.0e9)
    assert result[4, 4] != pytest.approx(-1.0e9)
    assert result[3, 3] == pytest.approx(5.0, abs=1.0)
    assert result[4, 4] == pytest.approx(5.0, abs=1.0)


# ------------------------------ compute_diffraction_summary ------------------------------


def test_summary_single_array_no_bad_pixels() -> None:
    """Frame stats match direct numpy reductions when no bad pixels are present."""
    rng = numpy.random.default_rng(0)
    patterns = rng.integers(0, 100, size=(5, 4, 6), dtype=numpy.uint16)
    indexes = numpy.arange(5, dtype=numpy.intp)
    array = SimpleDiffractionArray('a', indexes, patterns)
    dataset = _make_dataset([array], (4, 6))

    summary = compute_diffraction_summary(dataset)

    assert isinstance(summary, DiffractionSummary)
    numpy.testing.assert_array_equal(summary.minimum_pattern, patterns.min(axis=0))
    numpy.testing.assert_array_equal(summary.maximum_pattern, patterns.max(axis=0))
    numpy.testing.assert_allclose(summary.mean_pattern, patterns.mean(axis=0))
    numpy.testing.assert_array_equal(summary.indexes, indexes)
    numpy.testing.assert_array_equal(summary.total_counts, patterns.sum(axis=(-2, -1)))


def test_summary_concatenates_indexes_and_totals_across_arrays() -> None:
    """indexes and total_counts are laid out in array order with the right total length."""
    patterns_a = numpy.ones((3, 4, 5), dtype=numpy.uint16)
    patterns_b = numpy.full((2, 4, 5), 7, dtype=numpy.uint16)
    indexes_a = numpy.array([10, 11, 12], dtype=numpy.intp)
    indexes_b = numpy.array([20, 21], dtype=numpy.intp)
    array_a = SimpleDiffractionArray('a', indexes_a, patterns_a)
    array_b = SimpleDiffractionArray('b', indexes_b, patterns_b)
    dataset = _make_dataset([array_a, array_b], (4, 5))

    summary = compute_diffraction_summary(dataset)

    assert summary.indexes.shape == (5,)
    numpy.testing.assert_array_equal(summary.indexes, [10, 11, 12, 20, 21])
    numpy.testing.assert_array_equal(summary.total_counts, [20, 20, 20, 140, 140])


def test_summary_total_counts_excludes_bad_pixels() -> None:
    """Per-pattern totals sum over good pixels only."""
    patterns = numpy.ones((3, 4, 5), dtype=numpy.uint16)
    patterns[:, 0, 0] = 999  # will be masked
    patterns[:, 3, 4] = 555  # will be masked
    indexes = numpy.arange(3, dtype=numpy.intp)
    array = SimpleDiffractionArray('a', indexes, patterns)
    bad = numpy.zeros((4, 5), dtype=numpy.bool_)
    bad[0, 0] = True
    bad[3, 4] = True
    dataset = _make_dataset([array], (4, 5), bad_pixels=bad)

    summary = compute_diffraction_summary(dataset)

    good = numpy.logical_not(bad)
    expected = patterns[:, good].sum(axis=-1)
    numpy.testing.assert_array_equal(summary.total_counts, expected)


def test_summary_inpaints_frame_arrays_at_bad_pixel_positions() -> None:
    """Frame min/mean/max at bad positions come from inpainting, not the raw reduction."""
    patterns = numpy.full((3, 8, 8), 10, dtype=numpy.int32)
    patterns[:, 3, 3] = 10_000_000  # would dominate max
    patterns[:, 4, 4] = -10_000_000  # would dominate min
    indexes = numpy.arange(3, dtype=numpy.intp)
    array = SimpleDiffractionArray('a', indexes, patterns)
    bad = numpy.zeros((8, 8), dtype=numpy.bool_)
    bad[3, 3] = True
    bad[4, 4] = True
    dataset = _make_dataset([array], (8, 8), bad_pixels=bad)

    summary = compute_diffraction_summary(dataset)

    # Good positions are unaffected by inpainting: exact per-pixel reduction.
    good = numpy.logical_not(bad)
    numpy.testing.assert_array_equal(summary.minimum_pattern[good], patterns.min(axis=0)[good])
    numpy.testing.assert_array_equal(summary.maximum_pattern[good], patterns.max(axis=0)[good])
    numpy.testing.assert_allclose(summary.mean_pattern[good], patterns.mean(axis=0)[good])

    # Bad positions do NOT keep the raw extreme values.
    assert summary.minimum_pattern[4, 4] != -10_000_000
    assert summary.maximum_pattern[3, 3] != 10_000_000
    # And are pulled toward the surrounding constant field of 10.
    assert summary.minimum_pattern[3, 3] == pytest.approx(10.0, abs=1.0)
    assert summary.maximum_pattern[3, 3] == pytest.approx(10.0, abs=1.0)
    assert summary.mean_pattern[3, 3] == pytest.approx(10.0, abs=1.0)


def test_summary_empty_dataset_returns_zero_frames_and_empty_arrays() -> None:
    """An empty dataset yields a well-formed summary at the detector extent."""
    dataset = _make_dataset([], (4, 6))

    summary = compute_diffraction_summary(dataset)

    assert summary.minimum_pattern.shape == (4, 6)
    assert summary.mean_pattern.shape == (4, 6)
    assert summary.maximum_pattern.shape == (4, 6)
    assert numpy.all(summary.minimum_pattern == 0)
    assert numpy.all(summary.mean_pattern == 0)
    assert numpy.all(summary.maximum_pattern == 0)
    assert summary.indexes.shape == (0,)
    assert summary.total_counts.shape == (0,)
