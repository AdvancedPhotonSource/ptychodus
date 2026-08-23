"""Tests for the total-counts pattern-drop filter.

Covers the free `compute_pattern_counts` helper (and its delegation from
`AssembledDiffractionData.get_pattern_counts`) plus the loader-level filter
in `LoadArray.__call__` that drops patterns whose good-pixel total counts
fall outside the (inclusive) bounds set on `DiffractionSettings`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy

from ptychodus.api.diffraction import (
    AssembledDiffractionData,
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
    compute_pattern_counts,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.preprocess.diffraction import DiffractionPrepStepUnion
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.diffraction.dataset import AssembledDiffractionDataset
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


# ---------- compute_pattern_counts ----------


def test_compute_pattern_counts_sums_only_good_pixels() -> None:
    patterns = numpy.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=numpy.int32,
    )
    bad_pixels = numpy.array([[False, True], [False, False]], dtype=bool)
    counts = compute_pattern_counts(patterns, bad_pixels)
    assert counts.tolist() == [1 + 3 + 4, 5 + 7 + 8]


def test_compute_pattern_counts_all_bad_returns_zero() -> None:
    patterns = numpy.ones((3, 2, 2), dtype=numpy.int32)
    bad_pixels = numpy.ones((2, 2), dtype=bool)
    counts = compute_pattern_counts(patterns, bad_pixels)
    assert counts.shape == (3,)
    assert numpy.all(counts == 0)


def test_compute_pattern_counts_shape_matches_pattern_count() -> None:
    patterns = numpy.arange(5 * 4 * 3, dtype=numpy.int32).reshape(5, 4, 3)
    bad_pixels = numpy.zeros((4, 3), dtype=bool)
    counts = compute_pattern_counts(patterns, bad_pixels)
    assert counts.shape == (5,)


def test_assembled_get_pattern_counts_delegates_to_free_helper() -> None:
    patterns = numpy.arange(3 * 2 * 2, dtype=numpy.int32).reshape(3, 2, 2)
    bad_pixels = numpy.array([[False, True], [False, False]], dtype=bool)
    data = AssembledDiffractionData(
        indexes=numpy.arange(3, dtype=numpy.intp),
        patterns=patterns,
        pixel_geometry=PixelGeometry(1.0, 1.0),
        bad_pixels=bad_pixels,
    )
    expected = compute_pattern_counts(patterns, bad_pixels)
    assert numpy.array_equal(data.get_pattern_counts(), expected)


# ---------- Loader-level integration ----------


def _make_dataset(
    total_counts_lower_bound: int | None = None,
    total_counts_upper_bound: int | None = None,
) -> AssembledDiffractionDataset:
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)

    if total_counts_lower_bound is not None:
        diffraction_settings.total_counts_lower_bound_enabled.set_value(True)
        diffraction_settings.total_counts_lower_bound.set_value(total_counts_lower_bound)
    if total_counts_upper_bound is not None:
        diffraction_settings.total_counts_upper_bound_enabled.set_value(True)
        diffraction_settings.total_counts_upper_bound.set_value(total_counts_upper_bound)

    sizer = PatternSizer(diffraction_settings)
    task_manager = MagicMock()
    task_monitor = MagicMock()

    dataset = AssembledDiffractionDataset(
        diffraction_settings,
        sizer,
        detector_settings,
        task_manager,
        task_monitor,
    )
    metadata = DiffractionMetadata(
        num_patterns_per_array=[0],
        pattern_dtype=numpy.dtype(numpy.int32),
        detector_extent=ImageExtent(width_px=2, height_px=2),
    )
    contents_tree = DiffractionDatasetLayoutNode.create_root()
    source = SimpleDiffractionDataset(metadata, contents_tree, [])
    dataset.reload(source)
    return dataset


def _make_array_with_known_counts() -> SimpleDiffractionArray:
    """Four 2x2 patterns with total counts 4, 10, 20, 40 (all pixels good)."""
    patterns = numpy.stack(
        [
            numpy.full((2, 2), 1, dtype=numpy.int32),  # sum = 4
            numpy.array([[3, 3], [2, 2]], dtype=numpy.int32),  # sum = 10
            numpy.full((2, 2), 5, dtype=numpy.int32),  # sum = 20
            numpy.full((2, 2), 10, dtype=numpy.int32),  # sum = 40
        ]
    )
    indexes = numpy.array([7, 8, 9, 10], dtype=numpy.intp)
    return SimpleDiffractionArray('test', indexes, patterns)


def _run_loader(dataset: AssembledDiffractionDataset) -> AssembledDiffractionData:
    array = _make_array_with_known_counts()
    captured: dict[str, AssembledDiffractionData] = {}

    def stub_assemble_array(array_index, label, data):  # type: ignore[no-untyped-def]
        captured['data'] = data

    dataset.assemble_array = stub_assemble_array  # type: ignore[method-assign]

    task = dataset.create_array_loader(array, process_patterns=True)
    task()
    return captured['data']


def test_loader_no_bounds_keeps_all_patterns() -> None:
    dataset = _make_dataset()
    data = _run_loader(dataset)
    assert data.get_patterns_shape() == (4, 2, 2)
    assert data.get_indexes().tolist() == [7, 8, 9, 10]


def test_loader_lower_bound_drops_below_and_keeps_at_boundary() -> None:
    dataset = _make_dataset(total_counts_lower_bound=10)
    data = _run_loader(dataset)
    # 4 dropped; 10, 20, 40 kept (10 at boundary is inclusive)
    assert data.get_indexes().tolist() == [8, 9, 10]
    assert data.get_pattern_counts().tolist() == [10, 20, 40]


def test_loader_upper_bound_drops_above_and_keeps_at_boundary() -> None:
    dataset = _make_dataset(total_counts_upper_bound=20)
    data = _run_loader(dataset)
    # 40 dropped; 4, 10, 20 kept (20 at boundary is inclusive)
    assert data.get_indexes().tolist() == [7, 8, 9]
    assert data.get_pattern_counts().tolist() == [4, 10, 20]


def test_loader_both_bounds_keeps_intersection() -> None:
    dataset = _make_dataset(total_counts_lower_bound=10, total_counts_upper_bound=20)
    data = _run_loader(dataset)
    assert data.get_indexes().tolist() == [8, 9]
    assert data.get_pattern_counts().tolist() == [10, 20]


def test_loader_all_dropped_yields_empty_data() -> None:
    dataset = _make_dataset(total_counts_lower_bound=100)
    data = _run_loader(dataset)
    assert data.get_patterns_shape() == (0, 2, 2)
    assert data.get_indexes().shape == (0,)


def test_loader_preserves_index_pattern_alignment_after_drop() -> None:
    """After the drop, indexes[i] must still name the pattern in patterns[i]."""
    dataset = _make_dataset(total_counts_lower_bound=10, total_counts_upper_bound=20)
    data = _run_loader(dataset)

    indexes = data.get_indexes()
    patterns = data.get_patterns()
    assert indexes.shape[0] == patterns.shape[0]
    # index 8 → pattern with sum 10; index 9 → pattern with sum 20
    assert indexes.tolist() == [8, 9]
    assert int(patterns[0].sum()) == 10
    assert int(patterns[1].sum()) == 20


def test_loader_disabled_toggle_ignores_bound_value() -> None:
    """Setting the bound value but leaving the enabled toggle False must be a no-op."""
    dataset = _make_dataset()
    # A bound value that WOULD drop every pattern, but the enabled toggle stays False.
    dataset._settings.total_counts_lower_bound.set_value(1000)  # type: ignore[attr-defined]
    data = _run_loader(dataset)
    assert data.get_indexes().tolist() == [7, 8, 9, 10]


# ---------- Pipeline invariance ----------


def test_prep_pipeline_has_no_counts_filter_step() -> None:
    """Guard against a future contributor moving this into the pattern pipeline.

    A DiffractionPrepStep that drops rows would silently break the indexes/patterns
    1:1 invariant that prepare_reconstruct_input relies on, because
    DiffractionPrepPipeline.__call__ passes the original indexes through unchanged.
    """
    step_names = {getattr(cls, '__name__', '') for cls in DiffractionPrepStepUnion.__args__}  # type: ignore[attr-defined]
    forbidden = {'FilterCountsStep', 'CountsFilterStep', 'DropPatternsByCountsStep'}
    assert step_names.isdisjoint(forbidden)
