"""Unit tests for online accumulation: FrameStatistics, DiffractionMomentAccumulator and friends.

Behaviors verified:
  - FrameStatistics matches numpy's min, mean and max for any chunking, merges exactly,
    and keeps its running mean at single-pattern scale over a long uint32 stream
  - the accumulator gives the same rows and frames for any chunking, order, thread
    interleaving or merge, including a merge from a pickled snapshot
  - snapshots are unaffected by later additions, and an empty accumulator gives None
  - the inpainted mean equals the mean of the inpainted frames
  - select, normalize_by_i0 and estimate_incident_photon_count on snapshots
  - compute_diffraction_moments applies carried detector corrections first, and is the last
    snapshot of iterate_diffraction_moments, whose earlier snapshots are of the prefixes
  - estimate_incident_beam: calibrated I0, uncalibrated I0 and no I0, each judged as
    documented, with an implausible calibration reported but kept
  - binned patterns are the prep pipeline's BinningStep, with a bin touching any bad pixel
    NaN, carried through every chunking, merge, selection and normalization
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
    estimate_incident_beam,
    iterate_diffraction_moments,
)
from ptychodus.api.assemble import estimate_incident_photon_count
from ptychodus.api.diffraction import BadPixels
from ptychodus.api.moments import FrameStatistics, compute_intensity_moments
from ptychodus.api.preprocess.diffraction import BinningStep, inpaint_bad_pixels

_SHAPE = (24, 32)
# 32 columns leave a partial bin of 2, which binning leaves out.
_BINNING = BinningStep(bin_size_x=3, bin_size_y=2)


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
        *_SHAPE, bad_pixels=_bad_pixels(), chunk_size=chunk_size, binning=_BINNING
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
        numpy.empty(0) if moments.binned_patterns is None else moments.binned_patterns[order],
    )


def _assert_same_moments(actual: DiffractionMoments, expected: DiffractionMoments) -> None:
    assert actual.num_patterns == expected.num_patterns
    numpy.testing.assert_array_equal(actual.minimum_frame, expected.minimum_frame)
    numpy.testing.assert_array_equal(actual.maximum_frame, expected.maximum_frame)
    numpy.testing.assert_allclose(actual.mean_frame, expected.mean_frame, rtol=1e-12)
    assert actual.binning == expected.binning

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
    accumulator = DiffractionMomentAccumulator(
        *_SHAPE, bad_pixels=_bad_pixels(), chunk_size=8, binning=_BINNING
    )

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
    first = DiffractionMomentAccumulator(
        *_SHAPE, bad_pixels=_bad_pixels(), chunk_size=16, binning=_BINNING
    )
    second = DiffractionMomentAccumulator(
        *_SHAPE, bad_pixels=_bad_pixels(), chunk_size=16, binning=_BINNING
    )
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


def test_merging_requires_the_same_binning() -> None:
    first = DiffractionMomentAccumulator(*_SHAPE, binning=BinningStep(bin_size_x=2, bin_size_y=2))
    second = DiffractionMomentAccumulator(*_SHAPE, binning=BinningStep(bin_size_x=2, bin_size_y=4))
    second.add(numpy.arange(3), _patterns(3))

    with pytest.raises(ValueError, match='same binning'):
        first.merge(second)

    with pytest.raises(ValueError, match='same binning'):
        DiffractionMomentAccumulator(*_SHAPE).merge(second)


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


def test_i0_removes_a_beam_decay(caplog: pytest.LogCaptureFixture) -> None:
    patterns = _patterns(10)
    decay = numpy.linspace(1.0, 0.5, 10)
    decayed = numpy.rint(patterns * decay[:, numpy.newaxis, numpy.newaxis]).astype(numpy.uint32)
    indexes = numpy.arange(10)
    reference = _accumulate(indexes, patterns, chunk_size=4, step=10)
    moments = _accumulate(indexes, decayed, chunk_size=4, step=10)

    normalized = moments.normalize_by_i0(indexes[::-1], 1000.0 * decay[::-1])

    ratio = normalized.total_counts / reference.total_counts
    numpy.testing.assert_allclose(ratio / ratio.mean(), 1.0, rtol=2e-3)

    counts = 1000.0 * decay
    counts[3] = 0.0

    with caplog.at_level(logging.WARNING, logger='ptychodus.api.accumulate'):
        partial = moments.normalize_by_i0(indexes[:-1], counts[:-1])

    assert partial.total_counts[3] == moments.total_counts[3]
    assert partial.total_counts[9] == moments.total_counts[9]
    assert any('2 I0 readings' in record.message for record in caplog.records)


def test_i0_indexes_must_be_unique() -> None:
    moments = _accumulate(numpy.arange(4), _patterns(4), chunk_size=4, step=4)

    with pytest.raises(ValueError, match='unique'):
        moments.normalize_by_i0(numpy.array([0, 0, 1, 2]), numpy.ones(4))


def test_incident_photon_count_is_a_high_quantile_of_the_totals() -> None:
    totals = numpy.concatenate([numpy.full(99, 100.0), [1.0e6]])

    assert estimate_incident_photon_count(totals, quantile=0.99) == pytest.approx(
        numpy.quantile(totals, 0.99)
    )
    assert estimate_incident_photon_count(totals, quantile=0.95) == pytest.approx(100.0)
    assert estimate_incident_photon_count(totals, quantile=1.0) == 1.0e6
    assert estimate_incident_photon_count(numpy.array([numpy.nan, 5.0]), quantile=1.0) == 5.0
    assert estimate_incident_photon_count(numpy.array([numpy.nan])) == 0.0

    with pytest.raises(ValueError):
        estimate_incident_photon_count(totals, quantile=1.5)


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


# ---------------------------------------------------------------------------
# Binned patterns
# ---------------------------------------------------------------------------


def test_binned_patterns_are_the_pipelines_binning_with_flagged_bins_nan() -> None:
    patterns = _patterns(5)
    bad_pixels = _bad_pixels()
    accumulator = DiffractionMomentAccumulator(*_SHAPE, bad_pixels=bad_pixels, binning=_BINNING)
    accumulator.add(numpy.arange(5), patterns)
    moments = accumulator.get_moments()
    assert moments is not None and moments.binned_patterns is not None

    binned = moments.binned_patterns
    expected = _BINNING.apply(numpy.where(bad_pixels, 0, patterns)).astype(numpy.float64)
    flagged = _BINNING.apply(bad_pixels)
    expected[:, flagged] = numpy.nan

    assert binned.shape == (5, 12, 10)
    assert binned.dtype == numpy.float32
    numpy.testing.assert_allclose(binned, expected, rtol=1e-6)
    # Bin (5, 5) holds the bad pixel (10, 15) among five good ones, and is flagged whole.
    assert flagged[5, 5]
    assert numpy.isnan(binned[:, 5, 5]).all()
    assert numpy.count_nonzero(flagged) == 1 + 4


def test_an_unbinned_accumulator_keeps_no_binned_patterns() -> None:
    accumulator = DiffractionMomentAccumulator(*_SHAPE)
    accumulator.add(numpy.arange(3), _patterns(3))
    moments = accumulator.get_moments()

    assert moments is not None
    assert moments.binning is None
    assert moments.binned_patterns is None


def test_a_binning_must_leave_a_whole_bin() -> None:
    with pytest.raises(ValueError, match='no whole bin'):
        DiffractionMomentAccumulator(*_SHAPE, binning=BinningStep(bin_size_x=33, bin_size_y=1))


def test_select_and_i0_carry_the_binned_patterns() -> None:
    patterns = _patterns(10)
    indexes = numpy.arange(10)
    moments = _accumulate(indexes, patterns, chunk_size=4, step=10)
    assert moments.binned_patterns is not None

    selected = moments.select([7, 2])
    assert selected.binned_patterns is not None
    numpy.testing.assert_array_equal(selected.binned_patterns, moments.binned_patterns[[2, 7]])

    i0 = numpy.linspace(1.0, 2.0, 10)
    normalized = moments.normalize_by_i0(indexes, i0)
    assert normalized.binned_patterns is not None
    scale = normalized.total_counts / moments.total_counts
    numpy.testing.assert_allclose(scale, i0.mean() / i0, rtol=1e-12)
    numpy.testing.assert_allclose(
        normalized.binned_patterns,
        moments.binned_patterns * scale[:, numpy.newaxis, numpy.newaxis],
        rtol=1e-6,
    )


def test_compute_diffraction_moments_bins_too() -> None:
    patterns = _patterns(30)
    indexes = numpy.arange(30)
    expected = _accumulate(indexes, patterns, chunk_size=30, step=30)

    moments = compute_diffraction_moments(
        indexes, patterns, bad_pixels=_bad_pixels(), chunk_size=7, binning=_BINNING
    )

    assert moments is not None
    _assert_same_moments(moments, expected)


def test_the_last_snapshot_is_the_whole_accumulation() -> None:
    patterns = _patterns(30)
    indexes = numpy.arange(30)
    expected = compute_diffraction_moments(
        indexes, patterns, bad_pixels=_bad_pixels(), chunk_size=7, binning=_BINNING
    )
    snapshots = list(
        iterate_diffraction_moments(
            indexes,
            patterns,
            bad_pixels=_bad_pixels(),
            chunk_size=7,
            binning=_BINNING,
            num_snapshots=4,
        )
    )

    assert expected is not None
    assert [num_seen for num_seen, _ in snapshots] == [8, 15, 22, 30]
    _assert_same_moments(snapshots[-1][1], expected)


def test_each_snapshot_is_the_accumulation_of_its_prefix() -> None:
    patterns = _patterns(20)
    indexes = numpy.arange(20)

    for num_seen, moments in iterate_diffraction_moments(
        indexes, patterns, bad_pixels=_bad_pixels(), chunk_size=3, num_snapshots=3
    ):
        expected = compute_diffraction_moments(
            indexes[:num_seen], patterns[:num_seen], bad_pixels=_bad_pixels(), chunk_size=3
        )
        assert expected is not None
        assert moments.num_patterns == num_seen
        numpy.testing.assert_array_equal(moments.indexes, expected.indexes)
        numpy.testing.assert_allclose(moments.total_counts, expected.total_counts)
        numpy.testing.assert_allclose(moments.mean_frame, expected.mean_frame)


def test_snapshots_never_repeat_and_need_at_least_one() -> None:
    patterns = _patterns(3)
    indexes = numpy.arange(3)
    stops = [
        num_seen
        for num_seen, _ in iterate_diffraction_moments(
            indexes, patterns, bad_pixels=_bad_pixels(), num_snapshots=10
        )
    ]

    assert stops == [1, 2, 3]
    assert not list(
        iterate_diffraction_moments(indexes[:0], patterns[:0], bad_pixels=_bad_pixels())
    )

    with pytest.raises(ValueError, match='snapshot'):
        next(
            iterate_diffraction_moments(
                indexes, patterns, bad_pixels=_bad_pixels(), num_snapshots=0
            )
        )


def _with_totals(totals: list[float]) -> DiffractionMoments:
    """Moments whose totals are exactly `totals`, at scan indexes 0, 1, ..."""
    num_patterns = len(totals)
    frame = numpy.zeros(_SHAPE)
    rows = numpy.zeros(num_patterns)
    return DiffractionMoments(
        num_patterns=num_patterns,
        minimum_frame=frame,
        mean_frame=frame,
        maximum_frame=frame,
        bad_pixels=numpy.zeros(_SHAPE, dtype=bool),
        indexes=numpy.arange(num_patterns),
        total_counts=numpy.asarray(totals, dtype=numpy.float64),
        center_of_mass_x_px=rows,
        center_of_mass_y_px=rows.copy(),
    )


def test_an_unattenuated_scan_is_transparent() -> None:
    estimate = estimate_incident_beam(_with_totals([64, 64, 64]))

    assert estimate.incident_photon_count == pytest.approx(64.0)
    assert estimate.mean_amplitude_transmission == pytest.approx(1.0)
    assert not estimate.decay_corrected


def test_without_i0_the_brightest_frames_stand_in_for_the_empty_beam() -> None:
    totals = [64.0, 32.0, 32.0, 32.0]
    estimate = estimate_incident_beam(_with_totals(totals))
    incident = numpy.quantile(totals, 0.99)

    assert estimate.incident_photon_count == pytest.approx(incident)
    assert estimate.mean_amplitude_transmission == pytest.approx(numpy.sqrt(40.0 / incident))
    assert estimate_incident_beam(
        _with_totals(totals), quantile=1.0
    ).mean_amplitude_transmission == pytest.approx(numpy.sqrt(40.0 / 64.0))


def test_one_anomalous_frame_does_not_set_the_incident_beam() -> None:
    totals = [32.0] * 150 + [64.0] * 49 + [640.0]
    estimate = estimate_incident_beam(_with_totals(totals))

    assert estimate.incident_photon_count == pytest.approx(64.0)
    assert estimate.mean_amplitude_transmission == pytest.approx(
        numpy.sqrt(numpy.mean(totals) / 64)
    )


def test_uncalibrated_i0_corrects_the_decay_before_the_quantile() -> None:
    """The beam halves for the second frame, so its half-height total is no absorption."""
    estimate = estimate_incident_beam(_with_totals([64.0, 32.0]), {0: 2.0, 1: 1.0})

    assert estimate.decay_corrected
    numpy.testing.assert_allclose(estimate.moments.total_counts, [48.0, 48.0])
    assert estimate.incident_photon_count == pytest.approx(48.0)
    assert estimate.mean_amplitude_transmission == pytest.approx(1.0)


def test_calibrated_i0_gives_the_incident_photons_and_a_per_position_transmission() -> None:
    estimate = estimate_incident_beam(
        _with_totals([64.0, 32.0]), {0: 32.0, 1: 32.0}, i0_photons_per_count=2.0
    )

    assert estimate.decay_corrected
    assert estimate.incident_photon_count == pytest.approx(64.0)
    # Per-position intensity transmissions are 1.0 and 0.5; their mean is 0.75.
    assert estimate.mean_amplitude_transmission == pytest.approx(numpy.sqrt(0.75))


def test_a_calibration_without_i0_is_unused() -> None:
    plain = estimate_incident_beam(_with_totals([64.0, 32.0]))
    calibrated = estimate_incident_beam(_with_totals([64.0, 32.0]), i0_photons_per_count=5.0)

    assert calibrated.incident_photon_count == pytest.approx(plain.incident_photon_count)
    assert calibrated.mean_amplitude_transmission == pytest.approx(
        plain.mean_amplitude_transmission
    )
    assert not calibrated.decay_corrected


def test_a_non_positive_calibration_leaves_i0_uncalibrated() -> None:
    uncalibrated = estimate_incident_beam(_with_totals([64.0, 32.0]), {0: 2.0, 1: 1.0})
    zero = estimate_incident_beam(
        _with_totals([64.0, 32.0]), {0: 2.0, 1: 1.0}, i0_photons_per_count=0.0
    )

    assert zero.incident_photon_count == pytest.approx(uncalibrated.incident_photon_count)


def test_positions_without_usable_i0_drop_out_of_a_calibrated_estimate() -> None:
    estimate = estimate_incident_beam(
        _with_totals([64.0, 32.0]), {0: 0.0, 1: 64.0}, i0_photons_per_count=1.0
    )

    assert estimate.incident_photon_count == pytest.approx(64.0)
    assert estimate.mean_amplitude_transmission == pytest.approx(numpy.sqrt(0.5))


def test_an_implausible_calibration_is_reported_but_kept(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger='ptychodus.api.accumulate'):
        estimate = estimate_incident_beam(
            _with_totals([64.0, 64.0]), {0: 32.0, 1: 32.0}, i0_photons_per_count=1.0
        )

    assert estimate.incident_photon_count == pytest.approx(32.0)
    assert estimate.mean_amplitude_transmission == pytest.approx(1.0)
    assert any('I0 calibration' in record.message for record in caplog.records)


def test_a_scan_that_recorded_nothing_is_transparent() -> None:
    estimate = estimate_incident_beam(_with_totals([0.0, 0.0]))

    assert estimate.incident_photon_count == 0.0
    assert estimate.mean_amplitude_transmission == 1.0


def test_transmission_bounds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match='minimum'):
        estimate_incident_beam(
            _with_totals([1.0]), min_intensity_transmission=2.0, max_intensity_transmission=1.0
        )
