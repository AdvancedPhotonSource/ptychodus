"""Unit tests for online accumulation: FrameStatistics, DiffractionMomentAccumulator and friends.

Behaviors verified:
  - FrameStatistics matches numpy's min, mean and max for any chunking, merges exactly,
    and keeps its running mean at single-pattern scale over a long uint32 stream
  - the accumulator gives the same rows and frames for any chunking, order, thread
    interleaving or merge, including a merge from a pickled snapshot
  - snapshots are unaffected by later additions, and an empty accumulator gives None
  - the inpainted mean equals the mean of the inpainted frames
  - select, normalize_by_monitor and estimate_incident_counts on snapshots
  - compute_diffraction_moments applies carried detector corrections first
"""

from concurrent.futures import ThreadPoolExecutor
import logging
import pickle

import numpy
import numpy.testing
import pytest

from ptychodus.api.accumulate import (
    DiffractionMomentAccumulator,
    DiffractionMoments,
    compute_diffraction_moments,
)
from ptychodus.api.assemble import estimate_incident_counts
from ptychodus.api.diffraction import BadPixels
from ptychodus.api.moments import FrameStatistics, compute_intensity_moments
from ptychodus.api.preprocess.diffraction import inpaint_bad_pixels

_SHAPE = (24, 32)


def _patterns(num_patterns: int, *, seed: int = 0) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    y = numpy.arange(_SHAPE[0]).reshape(-1, 1) - 11.0
    x = numpy.arange(_SHAPE[1]).reshape(1, -1) - 16.0
    shifts = rng.normal(0.0, 1.5, size=(num_patterns, 2))
    expected = numpy.stack(
        [1000.0 * numpy.exp(-((x - sx) ** 2 + (y - sy) ** 2) / 20.0) + 3.0 for sx, sy in shifts]
    )
    return rng.poisson(expected).astype(numpy.uint32)


def _bad_pixels() -> BadPixels:
    bad_pixels: BadPixels = numpy.zeros(_SHAPE, dtype=bool)
    bad_pixels[10, 15] = True
    bad_pixels[5:7, 20:22] = True
    return bad_pixels


def _accumulate(
    indexes: numpy.ndarray, patterns: numpy.ndarray, *, chunk_size: int, step: int
) -> DiffractionMoments:
    accumulator = DiffractionMomentAccumulator(
        *_SHAPE, bad_pixels=_bad_pixels(), chunk_size=chunk_size
    )

    for start in range(0, len(indexes), step):
        accumulator.add(indexes[start : start + step], patterns[start : start + step])

    moments = accumulator.get_moments()
    assert moments is not None
    return moments


def _sorted_rows(moments: DiffractionMoments) -> tuple[numpy.ndarray, ...]:
    order = numpy.argsort(moments.indexes)
    return (
        moments.indexes[order],
        moments.total_counts[order],
        moments.center_of_mass_x_px[order],
        moments.center_of_mass_y_px[order],
    )


def _assert_same_moments(actual: DiffractionMoments, expected: DiffractionMoments) -> None:
    assert actual.num_patterns == expected.num_patterns
    numpy.testing.assert_array_equal(actual.minimum_frame, expected.minimum_frame)
    numpy.testing.assert_array_equal(actual.maximum_frame, expected.maximum_frame)
    numpy.testing.assert_allclose(actual.mean_frame, expected.mean_frame, rtol=1e-12)

    for got, want in zip(_sorted_rows(actual), _sorted_rows(expected)):
        numpy.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# FrameStatistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('step', [1, 7, 40])
def test_frame_statistics_match_numpy_for_any_chunking(step: int) -> None:
    patterns = _patterns(40)
    statistics = FrameStatistics(*_SHAPE)

    for start in range(0, 40, step):
        statistics.add(patterns[start : start + step])

    frames = statistics.get_frames()
    assert frames is not None
    minimum_frame, mean_frame, maximum_frame = frames

    numpy.testing.assert_array_equal(minimum_frame, patterns.min(axis=0))
    numpy.testing.assert_array_equal(maximum_frame, patterns.max(axis=0))
    numpy.testing.assert_allclose(mean_frame, patterns.mean(axis=0, dtype=numpy.float64))
    assert minimum_frame.dtype == patterns.dtype
    assert mean_frame.dtype == numpy.float64


def test_frame_statistics_merge_equals_one_pass() -> None:
    patterns = _patterns(30)
    left, right, whole = (FrameStatistics(*_SHAPE) for _ in range(3))
    left.add(patterns[:11])
    right.add(patterns[11:])
    whole.add(patterns)

    left.merge(right)

    for merged, single in zip(left.get_frames() or (), whole.get_frames() or ()):
        numpy.testing.assert_allclose(merged, single, rtol=1e-12)

    assert left.num_patterns == 30


def test_running_mean_stays_at_pattern_scale_over_a_long_stream() -> None:
    rng = numpy.random.default_rng(4)
    statistics = FrameStatistics(4, 4)
    chunks = [
        rng.integers(4_000_000_000, 4_294_967_295, size=(1000, 4, 4), dtype=numpy.uint32)
        for _ in range(50)
    ]

    for chunk in chunks:
        statistics.add(chunk)

    frames = statistics.get_frames()
    assert frames is not None
    expected = numpy.mean(numpy.concatenate(chunks), axis=0, dtype=numpy.float64)
    numpy.testing.assert_allclose(frames[1], expected, rtol=1e-14)
    assert frames[1].max() < 4_294_967_295


def test_empty_frame_statistics_have_no_frames() -> None:
    statistics = FrameStatistics(*_SHAPE)
    statistics.add(numpy.zeros((0, *_SHAPE)))

    assert statistics.get_frames() is None
    assert FrameStatistics.from_frames(0, *(numpy.zeros(_SHAPE),) * 3).get_frames() is None


# ---------------------------------------------------------------------------
# DiffractionMomentAccumulator
# ---------------------------------------------------------------------------


def test_rows_are_the_inpainted_totals_and_centroids() -> None:
    patterns = _patterns(9)
    indexes = numpy.arange(100, 109)

    moments = _accumulate(indexes, patterns, chunk_size=4, step=3)

    repaired = inpaint_bad_pixels(patterns.astype(numpy.float64), _bad_pixels())
    expected = compute_intensity_moments(repaired)
    numpy.testing.assert_array_equal(moments.indexes, indexes)
    numpy.testing.assert_allclose(moments.total_counts, expected.total, rtol=1e-12)
    numpy.testing.assert_allclose(moments.center_of_mass_x_px, expected.centroid_x_px)
    numpy.testing.assert_allclose(moments.center_of_mass_y_px, expected.centroid_y_px)


@pytest.mark.parametrize(('chunk_size', 'step'), [(1, 1), (5, 3), (64, 17), (256, 64)])
def test_chunking_does_not_change_the_result(chunk_size: int, step: int) -> None:
    patterns = _patterns(64)
    indexes = numpy.arange(64)

    expected = _accumulate(indexes, patterns, chunk_size=64, step=64)
    actual = _accumulate(indexes, patterns, chunk_size=chunk_size, step=step)

    _assert_same_moments(actual, expected)


def test_order_and_threads_do_not_change_the_result() -> None:
    patterns = _patterns(60)
    indexes = numpy.arange(60)
    expected = _accumulate(indexes, patterns, chunk_size=8, step=60)
    permutation = numpy.random.default_rng(5).permutation(60)
    accumulator = DiffractionMomentAccumulator(*_SHAPE, bad_pixels=_bad_pixels(), chunk_size=8)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, 60, 3):
            rows = permutation[start : start + 3]
            executor.submit(accumulator.add, indexes[rows], patterns[rows])

    moments = accumulator.get_moments()
    assert moments is not None
    _assert_same_moments(moments, expected)


def test_merging_workers_equals_one_pass_also_through_a_pickled_snapshot() -> None:
    patterns = _patterns(50)
    indexes = numpy.arange(50)
    expected = _accumulate(indexes, patterns, chunk_size=16, step=50)
    first = DiffractionMomentAccumulator(*_SHAPE, bad_pixels=_bad_pixels(), chunk_size=16)
    second = DiffractionMomentAccumulator(*_SHAPE, bad_pixels=_bad_pixels(), chunk_size=16)
    first.add(indexes[:21], patterns[:21])
    second.add(indexes[21:], patterns[21:])

    snapshot = pickle.loads(pickle.dumps(second.get_moments()))
    first.merge(snapshot)

    moments = first.get_moments()
    assert moments is not None
    _assert_same_moments(moments, expected)


def test_merging_requires_the_same_bad_pixels() -> None:
    first = DiffractionMomentAccumulator(*_SHAPE, bad_pixels=_bad_pixels())
    second = DiffractionMomentAccumulator(*_SHAPE)
    second.add(numpy.arange(3), _patterns(3))

    with pytest.raises(ValueError, match='same bad pixels'):
        first.merge(second)


def test_a_snapshot_does_not_change_with_later_additions() -> None:
    patterns = _patterns(20)
    accumulator = DiffractionMomentAccumulator(*_SHAPE, chunk_size=6)
    accumulator.add(numpy.arange(10), patterns[:10])

    snapshot = accumulator.get_moments()
    accumulator.add(numpy.arange(10, 20), patterns[10:])
    later = accumulator.get_moments()

    assert snapshot is not None and later is not None
    assert snapshot.num_patterns == 10 and snapshot.indexes.size == 10
    assert later.num_patterns == 20 and later.indexes.size == 20


def test_an_empty_accumulator_has_no_moments() -> None:
    assert DiffractionMomentAccumulator(*_SHAPE).get_moments() is None


def test_mismatched_shapes_are_rejected() -> None:
    accumulator = DiffractionMomentAccumulator(*_SHAPE)

    with pytest.raises(ValueError, match='do not match'):
        accumulator.add(numpy.arange(2), numpy.zeros((2, 8, 8)))

    with pytest.raises(ValueError, match='one index per pattern'):
        accumulator.add(numpy.arange(3), numpy.zeros((2, *_SHAPE)))


# ---------------------------------------------------------------------------
# DiffractionMoments
# ---------------------------------------------------------------------------


def test_the_inpainted_mean_is_the_mean_of_the_inpainted_frames() -> None:
    patterns = _patterns(12)
    moments = _accumulate(numpy.arange(12), patterns, chunk_size=5, step=4)

    expected = numpy.mean(inpaint_bad_pixels(patterns.astype(numpy.float64), _bad_pixels()), axis=0)
    numpy.testing.assert_allclose(moments.mean_pattern(), expected, rtol=1e-10)


def test_select_keeps_the_rows_and_the_frames() -> None:
    moments = _accumulate(numpy.arange(10), _patterns(10), chunk_size=4, step=4)

    selected = moments.select([7, 2, 99])

    numpy.testing.assert_array_equal(selected.indexes, [2, 7])
    numpy.testing.assert_array_equal(selected.total_counts, moments.total_counts[[2, 7]])
    assert selected.num_patterns == moments.num_patterns
    numpy.testing.assert_array_equal(selected.mean_frame, moments.mean_frame)


def test_the_monitor_removes_a_beam_decay(caplog: pytest.LogCaptureFixture) -> None:
    patterns = _patterns(10)
    decay = numpy.linspace(1.0, 0.5, 10)
    decayed = numpy.rint(patterns * decay[:, numpy.newaxis, numpy.newaxis]).astype(numpy.uint32)
    indexes = numpy.arange(10)
    reference = _accumulate(indexes, patterns, chunk_size=4, step=10)
    moments = _accumulate(indexes, decayed, chunk_size=4, step=10)

    normalized = moments.normalize_by_monitor(indexes[::-1], 1000.0 * decay[::-1])

    ratio = normalized.total_counts / reference.total_counts
    numpy.testing.assert_allclose(ratio / ratio.mean(), 1.0, rtol=2e-3)

    counts = 1000.0 * decay
    counts[3] = 0.0

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.accumulate'):
        partial = moments.normalize_by_monitor(indexes[:-1], counts[:-1])

    assert partial.total_counts[3] == moments.total_counts[3]
    assert partial.total_counts[9] == moments.total_counts[9]
    assert any('2 incident photon counts' in record.message for record in caplog.records)


def test_monitor_indexes_must_be_unique() -> None:
    moments = _accumulate(numpy.arange(4), _patterns(4), chunk_size=4, step=4)

    with pytest.raises(ValueError, match='unique'):
        moments.normalize_by_monitor(numpy.array([0, 0, 1, 2]), numpy.ones(4))


def test_incident_counts_are_a_high_quantile_of_the_totals() -> None:
    totals = numpy.concatenate([numpy.full(99, 100.0), [1.0e6]])

    assert estimate_incident_counts(totals, quantile=0.99) == pytest.approx(
        numpy.quantile(totals, 0.99)
    )
    assert estimate_incident_counts(totals, quantile=0.95) == pytest.approx(100.0)
    assert estimate_incident_counts(totals, quantile=1.0) == 1.0e6
    assert estimate_incident_counts(numpy.array([numpy.nan, 5.0]), quantile=1.0) == 5.0
    assert estimate_incident_counts(numpy.array([numpy.nan])) == 0.0

    with pytest.raises(ValueError):
        estimate_incident_counts(totals, quantile=1.5)


# ---------------------------------------------------------------------------
# compute_diffraction_moments
# ---------------------------------------------------------------------------


def test_carried_corrections_are_applied_first() -> None:
    patterns = _patterns(8).astype(numpy.float64)
    background = numpy.full(_SHAPE, 3.0)
    flat_field = numpy.full(_SHAPE, 2.0)
    flat_field[0, 0] = numpy.nan
    raw = patterns * flat_field + background
    raw[:, 0, 0] = 1.0e9

    moments = compute_diffraction_moments(
        numpy.arange(8),
        raw,
        bad_pixels=_bad_pixels(),
        background=background,
        flat_field=flat_field,
        chunk_size=3,
    )
    expected = compute_diffraction_moments(
        numpy.arange(8), patterns, bad_pixels=_bad_pixels() | numpy.isnan(flat_field)
    )

    assert moments is not None and expected is not None
    assert moments.bad_pixels[0, 0]
    numpy.testing.assert_allclose(moments.total_counts, expected.total_counts, rtol=1e-9)
    numpy.testing.assert_allclose(
        moments.center_of_mass_x_px, expected.center_of_mass_x_px, atol=1e-9
    )


def test_an_empty_stack_has_no_moments() -> None:
    assert (
        compute_diffraction_moments(
            numpy.arange(0), numpy.zeros((0, *_SHAPE)), bad_pixels=_bad_pixels()
        )
        is None
    )
