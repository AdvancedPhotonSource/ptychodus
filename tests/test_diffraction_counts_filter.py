"""Tests for the total-counts pattern-drop filter.

Covers the free `compute_total_counts` helper (and its delegation from
`AssembledDiffractionData.get_total_counts`) plus the wiring that carries the
`DiffractionSettings` bounds into the assembly. The filter semantics themselves
(inclusive boundaries, index/pattern alignment, all-dropped) are pinned in
tests/test_assemble.py against the pure `preprocess_array`.

Also covers `compute_dataset_total_counts`, the pass that measures the counts the
filter actually compares its bounds against, and the guards that keep an array the
filter emptied from reaching the repository.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy
import pytest

from ptychodus.api.assemble import (
    AssembledDiffractionData,
    compute_dataset_total_counts,
    compute_total_counts,
)
from ptychodus.api.diffraction import (
    BadPixels,
    CropRegion,
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.preprocess.diffraction import DiffractionPrepStepUnion
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.diffraction.dataset import (
    AssembledDiffractionArray,
    AssembledDiffractionDataset,
)
from ptychodus.model.diffraction.monitor import DiffractionTaskMonitor
from ptychodus.model.diffraction.summary import DiffractionSummaryService
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings


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


def _load_dataset_with_bounds(
    lower: int | None = None,
    upper: int | None = None,
    *,
    enable_lower: bool = True,
    enable_upper: bool = True,
) -> AssembledDiffractionDataset:
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
    return dataset


def _load_with_bounds(
    lower: int | None = None,
    upper: int | None = None,
    *,
    enable_lower: bool = True,
    enable_upper: bool = True,
) -> AssembledDiffractionData:
    dataset = _load_dataset_with_bounds(
        lower, upper, enable_lower=enable_lower, enable_upper=enable_upper
    )
    return dataset.get_assembled_data()


def _known_counts_dataset(
    *, pixel_geometry: PixelGeometry | None = PixelGeometry(1.0, 1.0)
) -> SimpleDiffractionDataset:
    """The four known-counts patterns as a 4x4 detector dataset, one array.

    Pass ``pixel_geometry=None`` for the shape most readers produce: metadata
    that never had ``detector_pixel_geometry`` set.
    """
    patterns = numpy.zeros((4, 4, 4), dtype=numpy.int32)
    # Center 2x2 carries the known counts; the outer ring carries 100 per pixel,
    # so a crop to the center changes every total by a fixed, large amount.
    patterns[:, 1:3, 1:3] = _known_counts_array().get_patterns()
    patterns[:, 0, :] = 100
    patterns[:, 3, :] = 100
    patterns[:, :, 0] = 100
    patterns[:, :, 3] = 100
    indexes = numpy.array([7, 8, 9, 10], dtype=numpy.intp)
    array = SimpleDiffractionArray('test', indexes, patterns)
    metadata = DiffractionMetadata(
        num_patterns_per_array=[4],
        pattern_dtype=numpy.dtype(numpy.int32),
        detector_extent=ImageExtent(width_px=4, height_px=4),
        detector_pixel_geometry=pixel_geometry,
    )
    return SimpleDiffractionDataset(metadata, DiffractionDatasetLayoutNode.create_root(), [array])


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


# ---------- Emptied arrays ----------


def test_bounds_that_drop_everything_publish_no_array() -> None:
    """Regression: an emptied array used to reach the tree and crash on .max().

    A zero-pattern array carries no counts, no frames, and no mean pattern, so it
    is dropped rather than published. Its buffer slots keep their sentinel indexes.
    """
    dataset = _load_dataset_with_bounds(lower=1_000_000)
    assert len(dataset) == 0
    assert dataset.get_assembled_data().get_patterns().shape == (0, 2, 2)


def test_emptied_dataset_reports_no_mean_pattern_rather_than_nan() -> None:
    dataset = _load_dataset_with_bounds(lower=1_000_000)
    assert dataset.get_mean_pattern() is None


def test_empty_assembled_data_means_a_zero_frame_not_nan() -> None:
    """numpy.mean over an empty axis returns NaN; the accessor must not."""
    data = AssembledDiffractionData(
        indexes=numpy.zeros(0, dtype=numpy.intp),
        patterns=numpy.zeros((0, 3, 2), dtype=numpy.int32),
        pixel_geometry=PixelGeometry(1.0, 1.0),
        bad_pixels=numpy.zeros((3, 2), dtype=bool),
    )
    mean_pattern = data.get_mean_pattern()
    assert mean_pattern.shape == (3, 2)
    assert numpy.all(mean_pattern == 0.0)


def test_empty_array_counts_accessors_return_zero() -> None:
    """`.max()` on a zero-size array raises; both reductions are guarded."""
    data = AssembledDiffractionData(
        indexes=numpy.zeros(0, dtype=numpy.intp),
        patterns=numpy.zeros((0, 2, 2), dtype=numpy.int32),
        pixel_geometry=PixelGeometry(1.0, 1.0),
        bad_pixels=numpy.zeros((2, 2), dtype=bool),
    )
    array = AssembledDiffractionArray(array_index=0, label='empty', data=data)
    assert array.get_max_total_counts() == 0
    assert array.get_mean_total_counts() == 0.0


# ---------- Dataset mean total counts ----------


def _load_unequal_arrays_dataset() -> AssembledDiffractionDataset:
    """The four known-counts patterns split across two arrays, 1 frame then 3."""
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)
    task_manager = _InlineTaskManager()
    dataset = AssembledDiffractionDataset(
        diffraction_settings,
        detector_settings,
        task_manager,  # type: ignore[arg-type]
        DiffractionTaskMonitor(task_manager),  # type: ignore[arg-type]
    )
    patterns = _known_counts_array().get_patterns()
    arrays = [
        SimpleDiffractionArray('one', numpy.array([10], dtype=numpy.intp), patterns[3:]),
        SimpleDiffractionArray('three', numpy.array([7, 8, 9], dtype=numpy.intp), patterns[:3]),
    ]
    metadata = DiffractionMetadata(
        num_patterns_per_array=[1, 3],
        pattern_dtype=numpy.dtype(numpy.int32),
        detector_extent=ImageExtent(width_px=2, height_px=2),
    )
    source = SimpleDiffractionDataset(metadata, DiffractionDatasetLayoutNode.create_root(), arrays)
    dataset.reload(source)
    dataset.load_all_arrays(process_patterns=True, block=True)
    return dataset


def test_dataset_mean_total_counts_weights_arrays_by_frame_count() -> None:
    """Averaging the per-array means unweighted would answer 25.67 instead of 18.5."""
    dataset = _load_unequal_arrays_dataset()
    assert [array.get_num_patterns() for array in dataset] == [1, 3]
    assert dataset[0].get_mean_total_counts() == pytest.approx(40.0)
    assert dataset[1].get_mean_total_counts() == pytest.approx(34.0 / 3.0)
    # (40 + 4 + 10 + 20) / 4
    assert dataset.get_mean_total_counts() == pytest.approx(18.5)


def test_dataset_mean_total_counts_ignores_how_frames_are_grouped() -> None:
    """The same four frames in one array must answer the same as split across two."""
    assert _load_dataset_with_bounds().get_mean_total_counts() == pytest.approx(18.5)


def test_emptied_dataset_reports_zero_mean_total_counts() -> None:
    """numpy would divide by a zero frame count; the accessor must not."""
    dataset = _load_dataset_with_bounds(lower=1_000_000)
    assert len(dataset) == 0
    assert dataset.get_mean_total_counts() == 0.0


# ---------- compute_dataset_total_counts ----------


def test_dataset_total_counts_match_the_raw_totals_without_a_plan() -> None:
    measured = compute_dataset_total_counts(_known_counts_dataset())
    assert measured.indexes.tolist() == [7, 8, 9, 10]
    # 12 ring pixels at 100 each, plus the known center totals.
    assert measured.total_counts.tolist() == [1204, 1210, 1220, 1240]


def test_dataset_total_counts_follow_the_read_region() -> None:
    """The point of the pass: a crop changes the distribution the bounds face."""
    read_region = CropRegion(x_range=(1, 3), y_range=(1, 3))
    measured = compute_dataset_total_counts(_known_counts_dataset(), read_region=read_region)
    assert measured.indexes.tolist() == [7, 8, 9, 10]
    assert measured.total_counts.tolist() == [4, 10, 20, 40]


def test_dataset_total_counts_exclude_bad_pixels() -> None:
    bad_pixels = numpy.zeros((4, 4), dtype=bool)
    bad_pixels[1, 1] = True
    read_region = CropRegion(x_range=(1, 3), y_range=(1, 3))
    measured = compute_dataset_total_counts(
        _known_counts_dataset(), bad_pixels=bad_pixels, read_region=read_region
    )
    # Drops the [1, 1] pixel of each center block: 1, 3, 5, 10 respectively.
    assert measured.total_counts.tolist() == [3, 7, 15, 30]


def test_dataset_total_counts_never_apply_the_filter() -> None:
    """The pass measures the distribution bounds are chosen from, so it cannot filter."""
    measured = compute_dataset_total_counts(_known_counts_dataset())
    assert measured.total_counts.size == 4


def test_dataset_total_counts_need_no_detector_pixel_geometry() -> None:
    """Regression: the pass keeps only per-pattern totals, so a dataset whose
    reader never set detector_pixel_geometry must still measure.

    Refresh Counts used to raise ValueError here for every generic HDF5/NPZ/TIFF
    file, while an actual load of the same dataset succeeded through the
    DetectorSettings fallback.
    """
    measured = compute_dataset_total_counts(_known_counts_dataset(pixel_geometry=None))
    assert measured.indexes.tolist() == [7, 8, 9, 10]
    assert measured.total_counts.tolist() == [1204, 1210, 1220, 1240]


def test_dataset_total_counts_report_progress_over_the_arrays() -> None:
    progress: list[tuple[int, int]] = []
    compute_dataset_total_counts(
        _known_counts_dataset(), on_progress=lambda i, n: progress.append((i, n))
    )
    assert progress[0] == (0, 1)
    assert progress[-1] == (1, 1)


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


# ---------- DiffractionSummaryService ----------


def _summary_service(
    source: SimpleDiffractionDataset,
) -> tuple[DiffractionSummaryService, DetectorSettings]:
    """A service over a one-dataset repository, running its tasks inline."""
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)
    task_manager = _InlineTaskManager()

    dataset = AssembledDiffractionDataset(
        diffraction_settings,
        detector_settings,
        task_manager,  # type: ignore[arg-type]
        DiffractionTaskMonitor(task_manager),  # type: ignore[arg-type]
    )
    dataset.reload(source)

    class _FakeAPI:
        """Only the two members DiffractionSummaryService reaches for."""

        def get_repository(self) -> list[AssembledDiffractionDataset]:
            return [dataset]

        def load_bad_pixels(self, file_path: object, file_type: object = None) -> BadPixels:
            raise FileNotFoundError(file_path)

    service = DiffractionSummaryService(
        task_manager,  # type: ignore[arg-type]
        _FakeAPI(),  # type: ignore[arg-type]
        detector_settings,
        diffraction_settings,
    )
    return service, detector_settings


def test_service_measures_counts_without_a_detector_pixel_geometry() -> None:
    """Regression for the reported Refresh Counts failure.

    The real load path resolves a missing metadata geometry through
    DetectorSettings; the counts pass never saw that fallback, so it raised for
    every dataset whose reader leaves detector_pixel_geometry unset.
    """
    service, _ = _summary_service(_known_counts_dataset(pixel_geometry=None))

    assert service.compute_total_counts(0, force=True)

    assert service.task_monitor.get_last_error() is None
    measured = service.get_last_total_counts()
    assert measured is not None
    assert measured.total_counts.tolist() == [1204, 1210, 1220, 1240]


def test_service_caches_counts_against_the_processing_settings() -> None:
    service, _ = _summary_service(_known_counts_dataset(pixel_geometry=None))
    assert service.compute_total_counts(0)
    assert not service.compute_total_counts(0)
    assert not service.is_total_counts_stale(0)


# ---------- Error attribution ----------


def test_counts_pass_labels_the_monitor_with_its_own_actor() -> None:
    service, _ = _summary_service(_known_counts_dataset(pixel_geometry=None))
    service.compute_total_counts(0, force=True)
    assert service.task_monitor.actor == 'Refresh Counts'


def test_summarize_pass_labels_the_monitor_with_its_own_actor() -> None:
    service, _ = _summary_service(_known_counts_dataset(pixel_geometry=None))
    service.compute_total_counts(0, force=True)
    service.compute(0)
    assert service.task_monitor.actor == 'Compute Summary'


def test_a_failing_pass_leaves_the_error_and_the_actor_describing_it() -> None:
    """The two passes share one monitor, so a stale label would misattribute.

    The error here is raised inside the background task, which is the case that
    misreported: the monitor captures it and an observer replays it later, long
    after the ``except`` block that would have named the pass is gone.
    """
    # Metadata accounts for one array but the dataset holds two, so the counts
    # pass raises after entering the monitor.
    good = _known_counts_dataset(pixel_geometry=None)
    source = SimpleDiffractionDataset(
        good.get_metadata(),
        DiffractionDatasetLayoutNode.create_root(),
        [good[0], good[0]],
    )
    service, _ = _summary_service(source)

    # The monitor's default label, and the one the dialog used to report for
    # every failure regardless of which pass raised.
    assert service.task_monitor.actor == 'Compute Summary'
    assert service.task_monitor.get_last_error() is None

    with pytest.raises(ValueError, match='more arrays than metadata'):
        service.compute_total_counts(0, force=True)

    assert service.task_monitor.actor == 'Refresh Counts'
    assert isinstance(service.task_monitor.get_last_error(), ValueError)
