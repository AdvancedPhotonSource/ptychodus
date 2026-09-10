"""Unit tests for inpaint_bad_pixels and summarize_dataset."""

from collections.abc import Sequence
import threading
import time

import numpy
import pytest

from ptychodus.api.assemble import DiffractionSummary, summarize_dataset
from ptychodus.api.preprocess.diffraction import inpaint_bad_pixels
from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionDatasetLayoutNode,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPatterns,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent


def _make_dataset(
    arrays: Sequence[DiffractionArray],
    frame_shape: tuple[int, int],
    bad_pixels: BadPixels | None = None,
    num_patterns_per_array: Sequence[int] | None = None,
    pattern_dtype: numpy.dtype | None = None,
) -> SimpleDiffractionDataset:
    """Assemble a SimpleDiffractionDataset from hand-built arrays."""
    height, width = frame_shape
    metadata = DiffractionMetadata(
        num_patterns_per_array=(
            [a.get_num_patterns() for a in arrays]
            if num_patterns_per_array is None
            else list(num_patterns_per_array)
        ),
        pattern_dtype=numpy.dtype(numpy.uint16) if pattern_dtype is None else pattern_dtype,
        detector_extent=ImageExtent(width_px=width, height_px=height),
    )
    contents_tree = DiffractionDatasetLayoutNode.create_root()
    return SimpleDiffractionDataset(metadata, contents_tree, arrays, bad_pixels)


def _array(label: str, first_index: int, num_patterns: int, fill: int) -> SimpleDiffractionArray:
    patterns = numpy.full((num_patterns, 4, 5), fill, dtype=numpy.int32)
    indexes = numpy.arange(first_index, first_index + num_patterns, dtype=numpy.intp)
    return SimpleDiffractionArray(label, indexes, patterns)


class _FailingArray(DiffractionArray):
    """An array whose read raises, to exercise the error and skip paths."""

    def __init__(self, label: str, error: BaseException, num_patterns: int = 2) -> None:
        self._label = label
        self._error = error
        self._num_patterns = num_patterns

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return numpy.arange(self._num_patterns, dtype=numpy.intp)

    def get_patterns(self) -> DiffractionPatterns:
        raise self._error

    def get_num_patterns(self) -> int:
        return self._num_patterns


class _BlockingArray(DiffractionArray):
    """Parks its read on `gate`, to hold a worker at a known point in the fan-out.

    `entered` is set once the read is parked, so a test can wait until a worker is
    definitely inside this array before acting on that fact. `release_delay_sec`
    then keeps the worker busy for a beat after the gate opens, giving whoever
    opened it time to finish before this worker moves on to the next array.
    """

    def __init__(
        self,
        label: str,
        first_index: int,
        fill: int,
        gate: threading.Event,
        *,
        release_delay_sec: float = 0.0,
    ) -> None:
        self._inner = _array(label, first_index, 2, fill)
        self._gate = gate
        self._release_delay_sec = release_delay_sec
        self.entered = threading.Event()
        self.timed_out = False

    def get_label(self) -> str:
        return self._inner.get_label()

    def get_indexes(self) -> DiffractionIndexes:
        return self._inner.get_indexes()

    def get_patterns(self) -> DiffractionPatterns:
        self.entered.set()

        # The timeout is a deadlock guard, not a synchronization mechanism; a test
        # whose gate never opens should fail on `timed_out` rather than hang.
        if not self._gate.wait(timeout=10.0):
            self.timed_out = True

        if self._release_delay_sec > 0.0:
            time.sleep(self._release_delay_sec)

        return self._inner.get_patterns()

    def get_num_patterns(self) -> int:
        return self._inner.get_num_patterns()


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


# ------------------------------ summarize_dataset ------------------------------


def test_summary_single_array_no_bad_pixels() -> None:
    """Frame stats match direct numpy reductions when no bad pixels are present."""
    rng = numpy.random.default_rng(0)
    patterns = rng.integers(0, 100, size=(5, 4, 6), dtype=numpy.uint16)
    indexes = numpy.arange(5, dtype=numpy.intp)
    array = SimpleDiffractionArray('a', indexes, patterns)
    dataset = _make_dataset([array], (4, 6))

    summary = summarize_dataset(dataset)

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

    summary = summarize_dataset(dataset)

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

    summary = summarize_dataset(dataset)

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

    summary = summarize_dataset(dataset)

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

    summary = summarize_dataset(dataset)

    assert summary.minimum_pattern.shape == (4, 6)
    assert summary.mean_pattern.shape == (4, 6)
    assert summary.maximum_pattern.shape == (4, 6)
    assert numpy.all(summary.minimum_pattern == 0)
    assert numpy.all(summary.mean_pattern == 0)
    assert numpy.all(summary.maximum_pattern == 0)
    assert summary.indexes.shape == (0,)
    assert summary.total_counts.shape == (0,)


# ------------------------------ summarize_dataset: alignment surface ------------------------------


def test_summary_bad_pixels_override_wins_over_the_dataset_mask() -> None:
    """An explicit mask replaces the dataset's own, exactly as in assemble_dataset."""
    patterns = numpy.ones((3, 4, 5), dtype=numpy.int32)
    patterns[:, 0, 0] = 999
    patterns[:, 1, 1] = 555
    array = SimpleDiffractionArray('a', numpy.arange(3, dtype=numpy.intp), patterns)
    dataset_mask = numpy.zeros((4, 5), dtype=numpy.bool_)
    dataset_mask[0, 0] = True
    override = numpy.zeros((4, 5), dtype=numpy.bool_)
    override[1, 1] = True
    dataset = _make_dataset([array], (4, 5), bad_pixels=dataset_mask)

    summary = summarize_dataset(dataset, bad_pixels=override)

    # 999 at [0,0] is counted (not masked by the override); 555 at [1,1] is not.
    expected = patterns[:, numpy.logical_not(override)].sum(axis=-1)
    numpy.testing.assert_array_equal(summary.total_counts, expected)


def test_summary_rejects_a_bad_pixels_override_of_the_wrong_shape() -> None:
    """The detector-extent check in _resolve_bad_pixels applies here too."""
    dataset = _make_dataset([_array('a', 0, 2, 1)], (4, 5))

    with pytest.raises(ValueError):
        summarize_dataset(dataset, bad_pixels=numpy.zeros((3, 3), dtype=numpy.bool_))


def test_summary_skips_a_missing_array_and_summarizes_the_rest() -> None:
    """FileNotFoundError is logged and skipped rather than aborting the run."""
    dataset = _make_dataset(
        [
            _array('a', 0, 2, 1),
            _FailingArray('b', FileNotFoundError('gone')),
            _array('c', 10, 2, 3),
        ],
        (4, 5),
    )

    summary = summarize_dataset(dataset)

    # 'b' reserves slots 2-3; they keep their sentinel and are elided.
    numpy.testing.assert_array_equal(summary.indexes, [0, 1, 10, 11])
    numpy.testing.assert_array_equal(summary.total_counts, [20, 20, 60, 60])
    numpy.testing.assert_array_equal(summary.mean_pattern, numpy.full((4, 5), 2.0))


def test_summary_reports_an_array_error_without_aborting() -> None:
    """A non-FileNotFoundError failure goes to on_array_error; the others still summarize."""
    dataset = _make_dataset(
        [_array('a', 0, 2, 1), _FailingArray('b', RuntimeError('bad read'))], (4, 5)
    )
    errors: list[tuple[int, str]] = list()

    summary = summarize_dataset(
        dataset, on_array_error=lambda index, label, ex: errors.append((index, label))
    )

    assert errors == [(1, 'b')]
    numpy.testing.assert_array_equal(summary.indexes, [0, 1])


def test_summary_reports_progress_from_zero_to_the_array_count() -> None:
    """on_progress fires once up front and once per completed array."""
    arrays = [_array(f'a{n}', 10 * n, 2, n + 1) for n in range(4)]
    dataset = _make_dataset(arrays, (4, 5))
    progress: list[tuple[int, int]] = list()

    summarize_dataset(dataset, on_progress=lambda done, total: progress.append((done, total)))

    assert progress[0] == (0, 4)
    assert progress[-1] == (4, 4)
    assert len(progress) == 5


def test_summary_should_stop_returns_a_partial_summary() -> None:
    """Cancellation yields a summary of whatever was read, not an exception."""
    # Cancellation can only take effect on arrays the pool has not started, so the
    # gated array holds the lone worker while the stop is requested: arrays 2..5 are
    # then guaranteed to be PENDING when _map_arrays sweeps future.cancel() over
    # them. should_stop has to release the gate as it answers -- the release cannot
    # wait for a later callback, because a cancelled future only wakes as_completed
    # once a worker dequeues it, and this worker is the only one there is. The
    # release delay then keeps that worker inside the gated read while the sweep,
    # microseconds of work on an already-running thread, lands ahead of it.
    gate = threading.Event()
    gated = _BlockingArray('a1', 10, 2, gate, release_delay_sec=0.05)
    arrays: list[DiffractionArray] = [_array(f'a{n}', 10 * n, 2, n + 1) for n in range(6)]
    arrays[1] = gated

    def stop_once_the_worker_is_parked() -> bool:
        gated.entered.wait(timeout=10.0)
        gate.set()
        return True

    summary = summarize_dataset(
        _make_dataset(arrays, (4, 5)),
        should_stop=stop_once_the_worker_is_parked,
        max_workers=1,
    )

    assert not gated.timed_out
    # At least the first array landed; the pending ones were cancelled.
    assert 2 <= summary.indexes.size < 12
    assert summary.indexes.size == summary.total_counts.size


def test_summary_is_deterministic_regardless_of_worker_count() -> None:
    """Slot placement keeps results in array order however the futures complete."""
    arrays = [_array(f'a{n}', 100 * n, 3, n + 1) for n in range(5)]
    serial = summarize_dataset(_make_dataset(arrays, (4, 5)), max_workers=1)
    parallel = summarize_dataset(_make_dataset(arrays, (4, 5)), max_workers=4)

    numpy.testing.assert_array_equal(serial.indexes, parallel.indexes)
    numpy.testing.assert_array_equal(serial.total_counts, parallel.total_counts)
    numpy.testing.assert_array_equal(serial.minimum_pattern, parallel.minimum_pattern)
    numpy.testing.assert_array_equal(serial.maximum_pattern, parallel.maximum_pattern)
    numpy.testing.assert_allclose(serial.mean_pattern, parallel.mean_pattern)


def test_summary_rejects_more_arrays_than_metadata_accounts_for() -> None:
    """Same up-front validation assemble_dataset performs."""
    arrays = [_array('a', 0, 2, 1), _array('b', 10, 2, 3)]
    dataset = _make_dataset(arrays, (4, 5), num_patterns_per_array=[2])

    with pytest.raises(ValueError):
        summarize_dataset(dataset)
