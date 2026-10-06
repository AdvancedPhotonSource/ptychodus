from pathlib import Path

"""Unit tests for ptychodus.api.assemble.

Covers the pure assembly layer: buffer allocation, the per-array preprocessing
step, and the threaded fan-out over a whole dataset. No ptychodus.model import --
pipelines are built by hand rather than derived from PrepPipelineBuilder.
"""

from collections.abc import Callable
from typing import cast
import concurrent.futures
import os
import threading

import numpy
import pytest

from ptychodus.api.assemble import (
    AssembledDiffractionData,
    allocate_assembled_data,
    assemble_dataset,
    compute_array_offsets,
    compute_assembled_patterns_shape,
    default_max_workers,
    preprocess_array,
)
from ptychodus.api.diffraction import (
    BeamCenter,
    CropRegion,
    DiffractionArray,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.preprocess.diffraction import (
    BinningStep,
    DiffractionPrepPipeline,
    TransposeStep,
)

from diffraction_dataset_fixtures import (
    GEOMETRY,
    BlockingArray,
    FailingArray,
    make_array,
    make_dataset,
)

FRAME_SHAPE = (4, 4)


def _array(label: str, first_index: int, num_patterns: int, fill: int) -> SimpleDiffractionArray:
    return make_array(label, first_index, num_patterns, fill, frame_shape=FRAME_SHAPE)


# ---------- compute_array_offsets ----------


def test_offsets_are_cumulative_and_include_the_total() -> None:
    metadata = make_dataset([], (2, 2), num_patterns_per_array=[3, 0, 5]).get_metadata()
    assert list(compute_array_offsets(metadata)) == [0, 3, 3, 8]


def test_offsets_of_an_empty_dataset_are_just_zero() -> None:
    metadata = make_dataset([], (2, 2)).get_metadata()
    assert list(compute_array_offsets(metadata)) == [0]


# ---------- compute_assembled_patterns_shape ----------


def test_shape_without_a_pipeline_is_the_detector_extent() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    assert compute_assembled_patterns_shape(dataset) == (3, 4, 4)


def test_shape_reflects_the_pipeline_output() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    pipeline = DiffractionPrepPipeline(steps=(BinningStep(bin_size_x=2, bin_size_y=2),))
    assert compute_assembled_patterns_shape(dataset, pipeline) == (3, 2, 2)


def test_shape_honors_an_explicit_bad_pixels_override() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    override = numpy.zeros((4, 4), dtype=numpy.bool_)
    override[0, 0] = True
    assert compute_assembled_patterns_shape(dataset, bad_pixels=override) == (3, 4, 4)


def test_bad_pixels_shape_must_match_the_detector_extent() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    override = numpy.zeros((8, 8), dtype=numpy.bool_)

    with pytest.raises(ValueError, match='detector extent'):
        compute_assembled_patterns_shape(dataset, bad_pixels=override)


# ---------- allocate_assembled_data ----------


def test_allocation_prefills_indexes_with_the_sentinel() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    data = allocate_assembled_data(dataset)

    assert data.get_patterns_shape() == (3, 4, 4)
    assert data.get_patterns_dtype() == numpy.dtype(numpy.int32)
    assert data.get_indexes().size == 0
    assert data.get_patterns().shape == (0, 4, 4)


def test_allocation_uses_a_supplied_buffer_in_place_without_zeroing_it() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    buffer = numpy.full((3, 4, 4), 99, dtype=numpy.int32)

    data = allocate_assembled_data(dataset, patterns=buffer)

    assert numpy.shares_memory(data.get_pattern(0), buffer)
    assert numpy.all(buffer == 99)


def test_allocation_rejects_a_wrongly_shaped_buffer() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))

    with pytest.raises(ValueError, match='wrong shape'):
        allocate_assembled_data(dataset, patterns=numpy.zeros((3, 8, 8), dtype=numpy.int32))


def test_allocation_rejects_a_read_only_buffer() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    buffer = numpy.zeros((3, 4, 4), dtype=numpy.int32)
    buffer.flags.writeable = False

    with pytest.raises(ValueError, match='read-only'):
        allocate_assembled_data(dataset, patterns=buffer)


def test_allocation_honors_an_explicit_dtype_override() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    data = allocate_assembled_data(dataset, dtype=numpy.dtype(numpy.int64))
    assert data.get_patterns_dtype() == numpy.dtype(numpy.int64)


def test_allocation_promotes_buffer_dtype_for_binning_pipeline() -> None:
    """A uint16 dataset binned 4x4 needs at least uint32 headroom (65535*16 = 1048560)."""
    patterns = numpy.full((3, 4, 4), 1, dtype=numpy.uint16)
    array = SimpleDiffractionArray('a', numpy.arange(3, dtype=numpy.intp), patterns)
    dataset = make_dataset([array], (4, 4))
    pipeline = DiffractionPrepPipeline(steps=(BinningStep(bin_size_x=4, bin_size_y=4),))

    data = allocate_assembled_data(dataset, pipeline)

    patterns_dtype = data.get_patterns_dtype()

    assert numpy.issubdtype(patterns_dtype, numpy.unsignedinteger)
    # cast: the assert above is what makes this an integer dtype, and iinfo only
    # accepts one -- mypy cannot carry that narrowing across issubdtype.
    assert numpy.iinfo(cast('numpy.dtype[numpy.unsignedinteger]', patterns_dtype)).max >= 65535 * 16


# ---------- processed pixel geometry ----------


def test_geometry_is_unchanged_without_a_pipeline() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    assert allocate_assembled_data(dataset).get_pixel_geometry() == GEOMETRY


def test_geometry_is_multiplied_by_the_bin_size() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    pipeline = DiffractionPrepPipeline(steps=(BinningStep(bin_size_x=2, bin_size_y=4),))

    geometry = allocate_assembled_data(dataset, pipeline).get_pixel_geometry()

    assert geometry.width_m == pytest.approx(2 * GEOMETRY.width_m)
    assert geometry.height_m == pytest.approx(4 * GEOMETRY.height_m)


def test_geometry_is_swapped_by_a_transpose() -> None:
    dataset = make_dataset(
        [_array('a', 0, 3, 1)], (4, 4), pixel_geometry=PixelGeometry(width_m=1e-6, height_m=2e-6)
    )
    pipeline = DiffractionPrepPipeline(steps=(TransposeStep(),))

    geometry = allocate_assembled_data(dataset, pipeline).get_pixel_geometry()

    assert geometry == PixelGeometry(width_m=2e-6, height_m=1e-6)


def test_explicit_raw_geometry_overrides_the_metadata() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4))
    override = PixelGeometry(width_m=1e-3, height_m=2e-3)
    data = allocate_assembled_data(dataset, raw_pixel_geometry=override)
    assert data.get_pixel_geometry() == override


def test_missing_geometry_raises_rather_than_defaulting_to_zero() -> None:
    dataset = make_dataset([_array('a', 0, 3, 1)], (4, 4), pixel_geometry=None)

    with pytest.raises(ValueError, match='pixel geometry'):
        allocate_assembled_data(dataset)


# ---------- AssembledDiffractionData.get_probe_photon_count ----------


def test_photon_count_is_the_brightest_pattern_total() -> None:
    patterns = numpy.stack(
        [
            numpy.full((4, 4), 1, dtype=numpy.int32),  # sum = 16
            numpy.full((4, 4), 5, dtype=numpy.int32),  # sum = 80
            numpy.full((4, 4), 2, dtype=numpy.int32),  # sum = 32
        ]
    )
    array = SimpleDiffractionArray('a', numpy.arange(3, dtype=numpy.intp), patterns)

    data = assemble_dataset(make_dataset([array], (4, 4)))

    assert data.get_probe_photon_count() == 80


def test_photon_count_excludes_bad_pixels() -> None:
    patterns = numpy.full((2, 4, 4), 3, dtype=numpy.int32)
    patterns[:, 0, 0] = 1000  # masked, must not dominate
    array = SimpleDiffractionArray('a', numpy.arange(2, dtype=numpy.intp), patterns)
    bad = numpy.zeros((4, 4), dtype=numpy.bool_)
    bad[0, 0] = True

    data = assemble_dataset(make_dataset([array], (4, 4), bad_pixels=bad))

    assert data.get_probe_photon_count() == 45  # 15 good pixels x 3


def test_photon_count_of_an_unfilled_buffer_is_zero() -> None:
    """Regression: .max() on an empty reduction used to raise from the GUI button."""
    dataset = make_dataset([], (4, 4), num_patterns_per_array=[4])

    assert allocate_assembled_data(dataset).get_probe_photon_count() == 0
    assert AssembledDiffractionData.create_null().get_probe_photon_count() == 0


# ---------- preprocess_array ----------


def _counts_array() -> SimpleDiffractionArray:
    """Four 2x2 patterns with total counts 4, 10, 20, 40 (all pixels good)."""
    patterns = numpy.stack(
        [
            numpy.full((2, 2), 1, dtype=numpy.int32),  # sum = 4
            numpy.array([[3, 3], [2, 2]], dtype=numpy.int32),  # sum = 10
            numpy.full((2, 2), 5, dtype=numpy.int32),  # sum = 20
            numpy.full((2, 2), 10, dtype=numpy.int32),  # sum = 40
        ]
    )
    return SimpleDiffractionArray('test', numpy.array([7, 8, 9, 10], dtype=numpy.intp), patterns)


def _preprocess_counts(
    lower: int | None = None, upper: int | None = None
) -> AssembledDiffractionData:
    good = numpy.zeros((2, 2), dtype=numpy.bool_)
    return preprocess_array(
        _counts_array(),
        raw_bad_pixels=good,
        processed_bad_pixels=good,
        raw_pixel_geometry=GEOMETRY,
        total_counts_lower_bound=lower,
        total_counts_upper_bound=upper,
    )


def test_bad_pixels_are_zeroed_in_cropped_coords_after_load_time_crop() -> None:
    """A saturated bad pixel inside the load-time crop must not survive the pipeline.

    With crop hoisted to load time, the reader delivers pre-cropped patterns and
    preprocess_array zeroes bad pixels against a matching cropped mask. Result must
    match the pre-hoist behavior of zero-in-raw-coords-then-crop.
    """
    full_patterns = numpy.zeros((2, 40, 60), dtype=numpy.uint16)
    full_patterns[:, 18, 30] = 65535
    full_bad = numpy.zeros((40, 60), dtype=numpy.bool_)
    full_bad[18, 30] = True

    read_region = CropRegion.from_center_extent(
        BeamCenter(x_px=30, y_px=18), ImageExtent(width_px=16, height_px=12)
    )
    array = SimpleDiffractionArray('a', numpy.arange(2, dtype=numpy.intp), full_patterns)
    cropped_bad = read_region.apply_to(full_bad)

    data = preprocess_array(
        array,
        raw_bad_pixels=cropped_bad,
        processed_bad_pixels=cropped_bad,
        raw_pixel_geometry=GEOMETRY,
        read_region=read_region,
    )

    assert data.get_patterns_shape() == (2, 12, 16)
    assert data.get_bad_pixels().shape == (12, 16)
    assert data.get_bad_pixels()[6, 8]
    assert numpy.all(data.get_patterns() == 0)


def test_preprocess_without_a_pipeline_keeps_raw_shapes_but_still_zeroes() -> None:
    patterns = numpy.full((2, 4, 4), 7, dtype=numpy.int32)
    array = SimpleDiffractionArray('a', numpy.arange(2, dtype=numpy.intp), patterns)
    raw_bad = numpy.zeros((4, 4), dtype=numpy.bool_)
    raw_bad[1, 1] = True

    data = preprocess_array(
        array, raw_bad_pixels=raw_bad, processed_bad_pixels=raw_bad, raw_pixel_geometry=GEOMETRY
    )

    assert data.get_patterns_shape() == (2, 4, 4)
    assert numpy.all(data.get_patterns()[:, 1, 1] == 0)
    assert numpy.all(data.get_patterns()[:, 0, 0] == 7)


def test_preprocess_propagates_a_missing_file() -> None:
    good = numpy.zeros((4, 4), dtype=numpy.bool_)

    with pytest.raises(FileNotFoundError):
        preprocess_array(
            FailingArray('gone', FileNotFoundError('nope')),
            raw_bad_pixels=good,
            processed_bad_pixels=good,
            raw_pixel_geometry=GEOMETRY,
        )


def test_counts_filter_is_a_noop_without_bounds() -> None:
    data = _preprocess_counts()
    assert data.get_patterns_shape()[0] == 4
    numpy.testing.assert_array_equal(data.get_indexes(), [7, 8, 9, 10])


def test_counts_lower_bound_is_inclusive() -> None:
    data = _preprocess_counts(lower=10)
    numpy.testing.assert_array_equal(data.get_indexes(), [8, 9, 10])


def test_counts_upper_bound_is_inclusive() -> None:
    data = _preprocess_counts(upper=20)
    numpy.testing.assert_array_equal(data.get_indexes(), [7, 8, 9])


def test_counts_bounds_keep_the_intersection() -> None:
    data = _preprocess_counts(lower=10, upper=20)
    numpy.testing.assert_array_equal(data.get_indexes(), [8, 9])


def test_counts_filter_can_drop_everything() -> None:
    data = _preprocess_counts(lower=1000)
    assert data.get_patterns_shape() == (0, 2, 2)
    assert data.get_indexes().size == 0


def test_counts_filter_keeps_indexes_aligned_with_patterns() -> None:
    data = _preprocess_counts(lower=10, upper=20)
    patterns = data.get_patterns()
    assert patterns.shape[0] == data.get_indexes().shape[0]
    assert patterns[0].sum() == 10
    assert patterns[1].sum() == 20


# ---------- assemble_dataset ----------


def _three_arrays() -> SimpleDiffractionDataset:
    """Arrays of length 2, 3, 1 with distinct fills and contiguous scan indexes."""
    return make_dataset([_array('a', 0, 2, 1), _array('b', 2, 3, 2), _array('c', 5, 1, 3)], (4, 4))


def test_assemble_lays_arrays_out_in_offset_order() -> None:
    data = assemble_dataset(_three_arrays())

    assert data.get_patterns_shape() == (6, 4, 4)
    numpy.testing.assert_array_equal(data.get_indexes(), [0, 1, 2, 3, 4, 5])
    numpy.testing.assert_array_equal(data.get_patterns()[:, 0, 0], [1, 1, 2, 2, 2, 3])


def test_assemble_reports_progress_and_each_assembled_array() -> None:
    assembled: list[tuple[int, str]] = []
    progress: list[tuple[int, int]] = []

    assemble_dataset(
        _three_arrays(),
        on_array_assembled=lambda i, label, view: assembled.append((i, label)),
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert sorted(assembled) == [(0, 'a'), (1, 'b'), (2, 'c')]
    assert progress[0] == (0, 3)
    assert progress[-1] == (3, 3)
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)


def test_assemble_places_arrays_by_offset_even_when_they_finish_out_of_order() -> None:
    # The gated array is arrays[0], so pattern_dtype has to be supplied: deriving it
    # would read the gated array on this thread, before the gate can ever be opened,
    # and burn the whole deadlock guard before the scenario starts.
    gate_first = threading.Event()
    slow = BlockingArray(_array('slow', 0, 2, 1), gate_first)
    dataset = make_dataset(
        [slow, _array('fast', 2, 2, 2)],
        (4, 4),
        num_patterns_per_array=[2, 2],
        pattern_dtype=numpy.int32,
    )

    def release_after_fast(array_index: int, label: str, view: AssembledDiffractionData) -> None:
        if label == 'fast':
            gate_first.set()

    data = assemble_dataset(dataset, on_array_assembled=release_after_fast, max_workers=4)

    # Without this the test passes on the degenerate path, where the gate never opened
    # and the read fell through on the timeout instead of on 'fast' finishing first.
    assert not slow.timed_out
    numpy.testing.assert_array_equal(data.get_indexes(), [0, 1, 2, 3])
    numpy.testing.assert_array_equal(data.get_patterns()[:, 0, 0], [1, 1, 2, 2])


def test_assemble_leaves_sentinel_holes_where_the_counts_filter_dropped_patterns() -> None:
    """A short array under-fills its slice; the next array's region is untouched."""
    patterns_a = numpy.stack(
        [numpy.full((4, 4), 1, dtype=numpy.int32), numpy.full((4, 4), 100, dtype=numpy.int32)]
    )
    array_a = SimpleDiffractionArray('a', numpy.array([0, 1], dtype=numpy.intp), patterns_a)
    dataset = make_dataset([array_a, _array('b', 2, 2, 2)], (4, 4))

    data = assemble_dataset(dataset, total_counts_upper_bound=100)

    # Pattern 1 of array 'a' totals 1600 and is dropped; its slot keeps the sentinel.
    numpy.testing.assert_array_equal(data.get_indexes(), [0, 2, 3])
    numpy.testing.assert_array_equal(data.get_patterns()[:, 0, 0], [1, 2, 2])


def test_assemble_skips_a_missing_array_without_reporting_an_error() -> None:
    errors: list[tuple[int, str]] = []
    progress: list[tuple[int, int]] = []
    dataset = make_dataset(
        [
            _array('a', 0, 2, 1),
            FailingArray('gone', FileNotFoundError('nope')),
            _array('c', 4, 2, 3),
        ],
        (4, 4),
        num_patterns_per_array=[2, 2, 2],
    )

    data = assemble_dataset(
        dataset,
        on_array_error=lambda i, label, exc: errors.append((i, label)),
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert errors == []
    assert progress[-1] == (3, 3)
    numpy.testing.assert_array_equal(data.get_indexes(), [0, 1, 4, 5])


def test_assemble_reports_other_errors_and_keeps_going() -> None:
    errors: list[tuple[int, str, str]] = []
    dataset = make_dataset(
        [_array('a', 0, 2, 1), FailingArray('bad', RuntimeError('boom')), _array('c', 4, 2, 3)],
        (4, 4),
        num_patterns_per_array=[2, 2, 2],
    )

    data = assemble_dataset(
        dataset, on_array_error=lambda i, label, exc: errors.append((i, label, str(exc)))
    )

    assert errors == [(1, 'bad', 'boom')]
    numpy.testing.assert_array_equal(data.get_indexes(), [0, 1, 4, 5])


def test_assemble_stops_early_when_asked() -> None:
    # Cancellation can only take effect on arrays the pool has not started, so the
    # gated array holds the lone worker while the stop is requested: arrays 2..7 are
    # then guaranteed to be PENDING when _map_arrays sweeps future.cancel() over
    # them. should_stop has to release the gate as it answers -- the release cannot
    # wait for a later callback, because a cancelled future only wakes as_completed
    # once a worker dequeues it, and this worker is the only one there is. The
    # release delay then keeps that worker inside the gated read while the sweep,
    # microseconds of work on an already-running thread, lands ahead of it.
    gate = threading.Event()
    gated = BlockingArray(_array('a1', 2, 2, 2), gate, release_delay_sec=0.05)
    arrays: list[DiffractionArray] = [_array(f'a{i}', 2 * i, 2, i + 1) for i in range(8)]
    arrays[1] = gated
    assembled: list[int] = []

    def stop_once_the_worker_is_parked() -> bool:
        gated.entered.wait(timeout=10.0)
        gate.set()
        return True

    assemble_dataset(
        make_dataset(arrays, (4, 4)),
        on_array_assembled=lambda i, label, view: assembled.append(i),
        should_stop=stop_once_the_worker_is_parked,
        max_workers=1,
    )

    assert not gated.timed_out
    assert 0 < len(assembled) < 8


def test_assemble_returns_the_supplied_buffer() -> None:
    dataset = _three_arrays()
    out = allocate_assembled_data(dataset)

    assert assemble_dataset(dataset, out=out) is out
    numpy.testing.assert_array_equal(out.get_indexes(), [0, 1, 2, 3, 4, 5])


def test_assemble_stamps_the_buffer_geometry_onto_every_view() -> None:
    dataset = _three_arrays()
    out = allocate_assembled_data(dataset, raw_pixel_geometry=PixelGeometry(1e-3, 2e-3))
    seen: list[PixelGeometry] = []

    assemble_dataset(
        dataset, out=out, on_array_assembled=lambda i, l, v: seen.append(v.get_pixel_geometry())
    )

    assert seen == [PixelGeometry(1e-3, 2e-3)] * 3


def test_assemble_into_a_memmap_yields_read_only_views_over_a_writable_base(tmp_path: Path) -> None:
    dataset = _three_arrays()
    buffer = numpy.memmap(tmp_path / 'scratch.npy', dtype=numpy.int32, mode='w+', shape=(6, 4, 4))
    out = allocate_assembled_data(dataset, patterns=buffer)
    views: list[AssembledDiffractionData] = []

    assemble_dataset(dataset, out=out, on_array_assembled=lambda i, l, v: views.append(v))
    buffer.flush()

    numpy.testing.assert_array_equal(out.get_patterns()[:, 0, 0], [1, 1, 2, 2, 2, 3])
    assert all(not v.get_pattern(0).flags.writeable for v in views)
    # Handing out read-only views must not downgrade the buffer they are taken over, or
    # the next scatter into it would fail.
    assert not out.get_patterns().flags.writeable
    assert buffer.flags.writeable


def test_assemble_rejects_more_arrays_than_metadata_accounts_for() -> None:
    dataset = make_dataset(
        [_array('a', 0, 2, 1), _array('b', 2, 2, 2)], (4, 4), num_patterns_per_array=[2]
    )

    with pytest.raises(ValueError, match='more arrays than metadata'):
        assemble_dataset(dataset)


def test_assemble_rejects_an_array_that_overflows_its_reserved_slice() -> None:
    errors: list[str] = []
    dataset = make_dataset(
        [_array('a', 0, 4, 1), _array('b', 4, 2, 2)], (4, 4), num_patterns_per_array=[2, 2]
    )

    assemble_dataset(dataset, on_array_error=lambda i, label, exc: errors.append(str(exc)))

    assert len(errors) == 1
    assert 'more patterns than metadata reserves' in errors[0]


def test_assemble_accepts_an_empty_array_list_with_reserved_capacity() -> None:
    """The streaming path reloads with no arrays and appends frames later."""
    dataset = make_dataset([], (4, 4), num_patterns_per_array=[2, 2])

    data = assemble_dataset(dataset)

    assert data.get_patterns_shape() == (4, 4, 4)
    assert data.get_indexes().size == 0


# ---------- probe_photon_counts on AssembledDiffractionData ----------


def _bare_data(
    num_patterns: int = 3,
    *,
    probe_photon_counts: numpy.ndarray | None = None,
    fill: int = 4,
) -> AssembledDiffractionData:
    patterns = numpy.full((num_patterns, 4, 4), fill, dtype=numpy.int32)
    return AssembledDiffractionData(
        indexes=numpy.arange(num_patterns, dtype=numpy.intp),
        patterns=patterns,
        pixel_geometry=GEOMETRY,
        bad_pixels=numpy.zeros((4, 4), dtype=numpy.bool_),
        probe_photon_counts=probe_photon_counts,
    )


def test_missing_probe_photon_counts_falls_back_to_total_counts() -> None:
    data = _bare_data(num_patterns=3, fill=4)  # total_counts = 4 * 16 = 64 per pattern

    assert not data.has_measured_probe_photon_counts()
    numpy.testing.assert_array_equal(data.get_probe_photon_counts(), [64, 64, 64])
    numpy.testing.assert_array_equal(data.get_probe_photon_counts(), data.get_total_counts())
    assert data.get_probe_photon_count() == 64


def test_measured_probe_photon_counts_take_priority_over_total_counts() -> None:
    measured = numpy.array([100.0, 500.0, 200.0], dtype=numpy.float64)
    data = _bare_data(num_patterns=3, fill=4, probe_photon_counts=measured)

    assert data.has_measured_probe_photon_counts()
    numpy.testing.assert_array_equal(data.get_probe_photon_counts(), measured)
    # Scalar backward-compat: max of measured, not max of total counts.
    assert data.get_probe_photon_count() == 500


def test_probe_photon_counts_wrong_ndim_is_rejected() -> None:
    with pytest.raises(ValueError, match='probe photon counts'):
        _bare_data(probe_photon_counts=numpy.zeros((3, 2), dtype=numpy.float64))


def test_probe_photon_counts_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match='probe photon counts'):
        _bare_data(num_patterns=3, probe_photon_counts=numpy.zeros(4, dtype=numpy.float64))


def test_probe_photon_counts_reject_non_finite() -> None:
    partial = numpy.array([100.0, numpy.nan, 200.0], dtype=numpy.float64)
    with pytest.raises(ValueError, match='finite and non-negative'):
        _bare_data(num_patterns=3, fill=4, probe_photon_counts=partial)


def test_probe_photon_counts_reject_negative() -> None:
    negative = numpy.array([100.0, -1.0, 200.0], dtype=numpy.float64)
    with pytest.raises(ValueError, match='finite and non-negative'):
        _bare_data(num_patterns=3, fill=4, probe_photon_counts=negative)


# ---------- Hz -> counts bridge through preprocess_array / assemble_dataset ----------


def _array_with_flux(
    label: str,
    first_index: int,
    num_patterns: int,
    fill: int,
    flux_hz: numpy.ndarray | None,
) -> SimpleDiffractionArray:
    patterns = numpy.full((num_patterns, 4, 4), fill, dtype=numpy.int32)
    indexes = numpy.arange(first_index, first_index + num_patterns, dtype=numpy.intp)
    return SimpleDiffractionArray(label, indexes, patterns, probe_photon_flux_Hz=flux_hz)


def test_preprocess_array_converts_flux_Hz_to_counts_using_exposure() -> None:  # noqa: N802
    flux_hz = numpy.array([1000.0, 2000.0, 500.0], dtype=numpy.float64)
    array = _array_with_flux('a', 0, 3, 4, flux_hz)
    good = numpy.zeros((4, 4), dtype=numpy.bool_)

    block = preprocess_array(
        array,
        raw_bad_pixels=good,
        processed_bad_pixels=good,
        raw_pixel_geometry=GEOMETRY,
        exposure_time_s=0.5,
    )

    assert block.has_measured_probe_photon_counts()
    numpy.testing.assert_array_equal(block.get_probe_photon_counts(), [500.0, 1000.0, 250.0])


def test_preprocess_array_drops_flux_when_exposure_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    flux_hz = numpy.array([1000.0, 2000.0], dtype=numpy.float64)
    array = _array_with_flux('a', 0, 2, 4, flux_hz)
    good = numpy.zeros((4, 4), dtype=numpy.bool_)

    with caplog.at_level('WARNING', logger='ptychodus.api.assemble'):
        block = preprocess_array(
            array,
            raw_bad_pixels=good,
            processed_bad_pixels=good,
            raw_pixel_geometry=GEOMETRY,
            exposure_time_s=None,
        )

    assert not block.has_measured_probe_photon_counts()
    assert 'exposure_time_s' in caplog.text


def test_preprocess_array_drops_flux_when_exposure_is_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    flux_hz = numpy.array([1000.0, 2000.0], dtype=numpy.float64)
    array = _array_with_flux('a', 0, 2, 4, flux_hz)
    good = numpy.zeros((4, 4), dtype=numpy.bool_)

    with caplog.at_level('WARNING', logger='ptychodus.api.assemble'):
        block = preprocess_array(
            array,
            raw_bad_pixels=good,
            processed_bad_pixels=good,
            raw_pixel_geometry=GEOMETRY,
            exposure_time_s=0.0,
        )

    assert not block.has_measured_probe_photon_counts()
    assert 'exposure_time_s' in caplog.text


def test_counts_filter_keeps_probe_photon_counts_aligned_with_patterns() -> None:
    good = numpy.zeros((2, 2), dtype=numpy.bool_)
    patterns = numpy.stack(
        [
            numpy.full((2, 2), 1, dtype=numpy.int32),  # sum = 4  (dropped)
            numpy.full((2, 2), 5, dtype=numpy.int32),  # sum = 20 (kept)
            numpy.full((2, 2), 10, dtype=numpy.int32),  # sum = 40 (kept)
        ]
    )
    flux_hz = numpy.array([100.0, 500.0, 200.0], dtype=numpy.float64)
    array = SimpleDiffractionArray(
        'a', numpy.array([7, 8, 9], dtype=numpy.intp), patterns, probe_photon_flux_Hz=flux_hz
    )

    block = preprocess_array(
        array,
        raw_bad_pixels=good,
        processed_bad_pixels=good,
        raw_pixel_geometry=GEOMETRY,
        exposure_time_s=1.0,
        total_counts_lower_bound=10,
    )

    numpy.testing.assert_array_equal(block.get_indexes(), [8, 9])
    numpy.testing.assert_array_equal(block.get_probe_photon_counts(), [500.0, 200.0])


def test_assemble_dataset_allocates_counts_buffer_only_when_flux_is_available() -> None:
    flux_hz = numpy.array([10.0, 20.0, 30.0], dtype=numpy.float64)

    with_flux = make_dataset([_array_with_flux('a', 0, 3, 4, flux_hz)], (4, 4), exposure_time_s=0.5)
    data = assemble_dataset(with_flux)

    assert data.has_measured_probe_photon_counts()
    numpy.testing.assert_array_equal(data.get_probe_photon_counts(), [5.0, 10.0, 15.0])

    without_flux = make_dataset([_array('a', 0, 3, 4)], (4, 4), exposure_time_s=0.5)
    data_no_flux = assemble_dataset(without_flux)

    assert not data_no_flux.has_measured_probe_photon_counts()
    # Fallback path is total counts (fill=4, 4x4 patterns => 64 per pattern).
    numpy.testing.assert_array_equal(data_no_flux.get_probe_photon_counts(), [64, 64, 64])


def test_assemble_dataset_uses_fallback_when_any_array_lacks_flux() -> None:
    flux_hz = numpy.array([10.0, 20.0], dtype=numpy.float64)
    a_with = _array_with_flux('a', 0, 2, 4, flux_hz)
    b_without = _array('b', 2, 2, 5)  # No flux -> no counts buffer reserved.

    dataset = make_dataset([a_with, b_without], (4, 4), exposure_time_s=0.5)

    data = assemble_dataset(dataset)

    # All-or-nothing: any array without flux -> no buffer -> fallback path used.
    assert not data.has_measured_probe_photon_counts()
    numpy.testing.assert_array_equal(data.get_probe_photon_counts(), data.get_total_counts())


class TestGetImageExtent:
    def test_matches_last_two_dims_of_the_stored_patterns(self) -> None:
        # A single 5x7 pattern to prove the width_px/height_px axis assignment.
        data = AssembledDiffractionData(
            indexes=numpy.zeros(1, dtype=numpy.intp),
            patterns=numpy.zeros((1, 5, 7), dtype=numpy.uint16),
            pixel_geometry=PixelGeometry(width_m=1e-6, height_m=1e-6),
            bad_pixels=numpy.zeros((5, 7), dtype=numpy.bool_),
        )

        extent = data.get_image_extent()

        assert extent == ImageExtent(width_px=7, height_px=5)


# ---------- read-only accessors and the pattern count ----------


def _fully_assembled() -> AssembledDiffractionData:
    """A buffer whose every slot holds a pattern."""
    return assemble_dataset(_three_arrays())


def _partly_assembled() -> AssembledDiffractionData:
    """A buffer with one array's slots left unfilled, so the getters have to gather."""
    dataset = make_dataset(
        [_array('a', 0, 2, 1), FailingArray('b', OSError('unreadable'), num_patterns=3)],
        (4, 4),
    )
    return assemble_dataset(dataset, on_array_error=lambda *_: None)


@pytest.mark.parametrize('assembled', [_fully_assembled, _partly_assembled])
def test_the_getters_never_return_a_writable_array(
    assembled: Callable[[], AssembledDiffractionData],
) -> None:
    # The contract must not depend on how full the buffer happens to be, or a caller that
    # works on a complete scan corrupts the shared buffer on a partial one.
    data = assembled()

    assert not data.get_patterns().flags.writeable
    assert not data.get_indexes().flags.writeable


def test_a_fully_assembled_buffer_is_returned_without_copying() -> None:
    data = _fully_assembled()

    # get_pattern() is a handle on the backing store, so sharing memory with it is the
    # observable form of "this did not copy the whole buffer".
    assert numpy.shares_memory(data.get_patterns(), data.get_pattern(0))


def test_a_partly_assembled_buffer_gathers_into_a_copy() -> None:
    data = _partly_assembled()

    assert not numpy.shares_memory(data.get_patterns(), data.get_pattern(0))


def test_taking_a_read_only_handle_leaves_the_buffer_fillable() -> None:
    dataset = _three_arrays()
    out = allocate_assembled_data(dataset)

    # Downgrading a view must not downgrade its base, or the next scatter would fail.
    out.get_patterns()
    out.get_indexes()
    assemble_dataset(dataset, out=out)

    numpy.testing.assert_array_equal(out.get_patterns()[:, 0, 0], [1, 1, 2, 2, 2, 3])


@pytest.mark.parametrize('assembled', [_fully_assembled, _partly_assembled])
def test_the_pattern_count_matches_what_the_getter_returns(
    assembled: Callable[[], AssembledDiffractionData],
) -> None:
    data = assembled()

    assert data.get_num_patterns() == data.get_patterns().shape[0]


def test_the_pattern_count_is_below_the_buffer_capacity_when_a_slot_went_unfilled() -> None:
    data = _partly_assembled()

    assert data.get_num_patterns() < data.get_patterns_shape()[0]


# ---------- how many arrays the fan-out holds in flight ----------


def _record_pool_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    """Capture the `max_workers` every ThreadPoolExecutor in the fan-out is built with."""
    sizes: list[int | None] = []
    real = concurrent.futures.ThreadPoolExecutor

    class _Recorder(real):  # type: ignore[valid-type,misc]
        def __init__(self, max_workers: int | None = None, **kwargs: object) -> None:
            sizes.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _Recorder)
    return sizes


def test_the_fan_out_is_bounded_when_the_caller_names_no_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each worker holds a whole array block resident, so an unbounded pool is a memory
    # multiplier. ThreadPoolExecutor's own default would be min(32, cpu_count + 4).
    monkeypatch.delenv('LOCAL_WORLD_SIZE', raising=False)
    sizes = _record_pool_sizes(monkeypatch)

    assemble_dataset(_three_arrays())

    assert len(sizes) == 1
    assert sizes[0] is not None and 1 <= sizes[0] <= 8


def test_processes_sharing_a_node_each_take_a_share_of_the_budget() -> None:
    # One process per local GPU would otherwise each size itself as though it had the
    # node to itself. A world this large pins the expected answer on any host.
    assert default_max_workers(local_world_size=64) == 1


def test_the_ceiling_bounds_the_pool_below_the_core_count() -> None:
    # The ceiling is what a caller raises to trade peak memory for throughput; a value
    # above the core count cannot, which is the whole point of taking the minimum.
    assert default_max_workers(max_arrays_in_flight=1, local_world_size=1) == 1
    assert default_max_workers(max_arrays_in_flight=1000, local_world_size=1) == (
        os.cpu_count() or 1
    )


def test_a_world_larger_than_the_budget_still_leaves_one_worker() -> None:
    # Integer division would otherwise hand back a pool of zero, which
    # ThreadPoolExecutor rejects outright.
    assert default_max_workers(max_arrays_in_flight=2, local_world_size=1000) == 1


def test_an_unreadable_local_world_size_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOCAL_WORLD_SIZE', 'oops')
    sizes = _record_pool_sizes(monkeypatch)

    data = assemble_dataset(_three_arrays())

    assert sizes[0] is not None and 1 <= sizes[0] <= 8
    numpy.testing.assert_array_equal(data.get_patterns()[:, 0, 0], [1, 1, 2, 2, 2, 3])


def test_the_launcher_supplies_the_divisor_when_the_caller_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '64')

    assert default_max_workers() == 1


def test_an_explicit_worker_count_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '64')
    sizes = _record_pool_sizes(monkeypatch)

    assemble_dataset(_three_arrays(), max_workers=3)

    assert sizes == [3]
