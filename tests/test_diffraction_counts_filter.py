"""Tests for the total-counts pattern-drop filter.

Covers the free `compute_total_counts` helper (and its delegation from
`AssembledDiffractionData.get_total_counts`) plus the wiring that carries the
`DiffractionSettings` bounds into the assembly. The filter semantics themselves
(inclusive boundaries, index/pattern alignment, all-dropped) are pinned in
tests/test_assemble.py against the pure `preprocess_array`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy

from ptychodus.api.assemble import AssembledDiffractionData, compute_total_counts
from ptychodus.api.diffraction import (
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.preprocess.diffraction import DiffractionPrepStepUnion
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.diffraction.dataset import AssembledDiffractionDataset
from ptychodus.model.diffraction.monitor import DiffractionTaskMonitor
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


# ---------- compute_total_counts ----------


def test_compute_total_counts_sums_only_good_pixels() -> None:
    patterns = numpy.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=numpy.int32,
    )
    bad_pixels = numpy.array([[False, True], [False, False]], dtype=bool)
    counts = compute_total_counts(patterns, bad_pixels)
    assert counts.tolist() == [1 + 3 + 4, 5 + 7 + 8]


def test_compute_total_counts_all_bad_returns_zero() -> None:
    patterns = numpy.ones((3, 2, 2), dtype=numpy.int32)
    bad_pixels = numpy.ones((2, 2), dtype=bool)
    counts = compute_total_counts(patterns, bad_pixels)
    assert counts.shape == (3,)
    assert numpy.all(counts == 0)


def test_compute_total_counts_shape_matches_pattern_count() -> None:
    patterns = numpy.arange(5 * 4 * 3, dtype=numpy.int32).reshape(5, 4, 3)
    bad_pixels = numpy.zeros((4, 3), dtype=bool)
    counts = compute_total_counts(patterns, bad_pixels)
    assert counts.shape == (5,)


def test_assembled_get_total_counts_delegates_to_free_helper() -> None:
    patterns = numpy.arange(3 * 2 * 2, dtype=numpy.int32).reshape(3, 2, 2)
    bad_pixels = numpy.array([[False, True], [False, False]], dtype=bool)
    data = AssembledDiffractionData(
        indexes=numpy.arange(3, dtype=numpy.intp),
        patterns=patterns,
        pixel_geometry=PixelGeometry(1.0, 1.0),
        bad_pixels=bad_pixels,
    )
    expected = compute_total_counts(patterns, bad_pixels)
    assert numpy.array_equal(data.get_total_counts(), expected)


# ---------- Settings-to-assembly wiring ----------


class _InlineTaskManager:
    """Runs queued tasks immediately, so loads complete before the call returns."""

    is_stopping = False
    background_queue_size = 0
    foreground_queue_size = 0

    def put_background_task(self, task: Callable[[], None]) -> None:
        task()

    def put_foreground_task(self, task: Callable[[], None]) -> None:
        task()


def _known_counts_array() -> SimpleDiffractionArray:
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


def _load_with_bounds(
    lower: int | None = None,
    upper: int | None = None,
    *,
    enable_lower: bool = True,
    enable_upper: bool = True,
) -> AssembledDiffractionData:
    """Assemble one known array end-to-end under the given settings."""
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)

    if lower is not None:
        diffraction_settings.total_counts_lower_bound_enabled.set_value(enable_lower)
        diffraction_settings.total_counts_lower_bound.set_value(lower)

    if upper is not None:
        diffraction_settings.total_counts_upper_bound_enabled.set_value(enable_upper)
        diffraction_settings.total_counts_upper_bound.set_value(upper)

    task_manager = _InlineTaskManager()
    dataset = AssembledDiffractionDataset(
        diffraction_settings,
        PatternSizer(diffraction_settings),
        detector_settings,
        task_manager,  # type: ignore[arg-type]
        DiffractionTaskMonitor(task_manager),  # type: ignore[arg-type]
    )
    array = _known_counts_array()
    metadata = DiffractionMetadata(
        num_patterns_per_array=[array.get_num_patterns()],
        pattern_dtype=numpy.dtype(numpy.int32),
        detector_extent=ImageExtent(width_px=2, height_px=2),
    )
    source = SimpleDiffractionDataset(metadata, DiffractionDatasetLayoutNode.create_root(), [array])
    dataset.reload(source)
    dataset.load_all_arrays(process_patterns=True, block=True)
    return dataset.get_assembled_data()


def test_no_bounds_keeps_all_patterns() -> None:
    assert _load_with_bounds().get_indexes().tolist() == [7, 8, 9, 10]


def test_enabled_lower_bound_reaches_the_filter() -> None:
    assert _load_with_bounds(lower=10).get_indexes().tolist() == [8, 9, 10]


def test_enabled_upper_bound_reaches_the_filter() -> None:
    assert _load_with_bounds(upper=20).get_indexes().tolist() == [7, 8, 9]


def test_both_bounds_keep_the_intersection() -> None:
    assert _load_with_bounds(lower=10, upper=20).get_indexes().tolist() == [8, 9]


def test_disabled_toggle_ignores_the_bound_value() -> None:
    """A bound value that would drop everything is inert while its toggle is False."""
    data = _load_with_bounds(lower=1000, enable_lower=False)
    assert data.get_indexes().tolist() == [7, 8, 9, 10]


def test_dropped_patterns_leave_sentinel_holes_in_the_buffer() -> None:
    """The buffer keeps full capacity; dropped slots stay invisible via the -1 sentinel."""
    data = _load_with_bounds(lower=10, upper=20)
    assert data.get_patterns_shape() == (4, 2, 2)
    assert data.get_patterns().shape == (2, 2, 2)


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
