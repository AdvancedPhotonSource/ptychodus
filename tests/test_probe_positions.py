"""Unit tests for the optional per-position photon-count field on ProbePosition.

ProbePositionSequence enforces an all-or-nothing invariant: every point supplies a
i0 or none do. A mix indicates a reader bug and is rejected at
construction with ValueError.
"""

from __future__ import annotations

import numpy
import pytest

from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence


class TestProbePositionPhotonCount:
    def test_default_is_none(self) -> None:
        point = ProbePosition(index=0, x_m=1.0, y_m=2.0)
        assert point.i0 is None

    def test_value_is_carried(self) -> None:
        point = ProbePosition(index=0, x_m=1.0, y_m=2.0, i0=7.5)
        assert point.i0 == pytest.approx(7.5)


class TestProbePositionSequencePhotonCounts:
    def test_absent_when_no_point_supplies_one(self) -> None:
        seq = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(3)])
        assert seq.get_i0() is None

    def test_present_when_every_point_supplies_one(self) -> None:
        seq = ProbePositionSequence(
            [
                ProbePosition(0, 0.0, 0.0, i0=10.0),
                ProbePosition(1, 1.0, 0.0, i0=20.0),
                ProbePosition(2, 2.0, 0.0, i0=30.0),
            ]
        )
        counts = seq.get_i0()
        assert counts is not None
        assert counts.shape == (3,)
        numpy.testing.assert_array_equal(counts, [10.0, 20.0, 30.0])

    def test_mixed_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='every point or none'):
            ProbePositionSequence(
                [
                    ProbePosition(0, 0.0, 0.0, i0=10.0),
                    ProbePosition(1, 1.0, 0.0),
                    ProbePosition(2, 2.0, 0.0, i0=30.0),
                ]
            )

    def test_negative_probe_photon_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='finite non-negative'):
            ProbePositionSequence(
                [
                    ProbePosition(0, 0.0, 0.0, i0=10.0),
                    ProbePosition(1, 1.0, 0.0, i0=-1.0),
                    ProbePosition(2, 2.0, 0.0, i0=30.0),
                ]
            )

    def test_non_finite_probe_photon_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='finite non-negative'):
            ProbePositionSequence(
                [
                    ProbePosition(0, 0.0, 0.0, i0=10.0),
                    ProbePosition(1, 1.0, 0.0, i0=float('nan')),
                    ProbePosition(2, 2.0, 0.0, i0=30.0),
                ]
            )

    def test_iteration_yields_populated_counts(self) -> None:
        seq = ProbePositionSequence(
            [
                ProbePosition(0, 0.0, 0.0, i0=10.0),
                ProbePosition(1, 1.0, 0.0, i0=20.0),
            ]
        )
        points = list(seq)
        assert points[0].i0 == pytest.approx(10.0)
        assert points[1].i0 == pytest.approx(20.0)

    def test_iteration_yields_none_when_no_counts(self) -> None:
        seq = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(2)])
        for point in seq:
            assert point.i0 is None

    def test_slice_preserves_counts(self) -> None:
        seq = ProbePositionSequence(
            [ProbePosition(i, float(i), 0.0, i0=float(10 + i)) for i in range(5)]
        )
        subset = seq[1:4]
        assert len(subset) == 3
        as_list = list(subset)
        assert as_list[0].index == 1
        assert as_list[0].i0 == pytest.approx(11.0)
        assert as_list[2].i0 == pytest.approx(13.0)

    def test_slice_stays_uniform_when_no_counts_supplied(self) -> None:
        seq = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(5)])
        subset = seq[1:4]
        assert subset.get_i0() is None

    def test_copy_preserves_counts_and_is_independent(self) -> None:
        seq = ProbePositionSequence(
            [
                ProbePosition(0, 0.0, 0.0, i0=10.0),
                ProbePosition(1, 1.0, 0.0, i0=20.0),
            ]
        )
        clone = seq.copy()
        original = seq.get_i0()
        cloned = clone.get_i0()
        assert original is not None
        assert cloned is not None
        assert cloned is not original
        numpy.testing.assert_array_equal(cloned, original)

    def test_nbytes_includes_counts(self) -> None:
        without = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(3)])
        with_counts = ProbePositionSequence(
            [ProbePosition(i, float(i), 0.0, i0=float(i)) for i in range(3)]
        )
        assert with_counts.nbytes > without.nbytes


class TestProbePositionSequenceInterpolateAtIndexes:
    def test_duplicates_average_gaps_interpolate_and_the_outside_is_left_out(self) -> None:
        seq = ProbePositionSequence(
            [
                ProbePosition(0, 0.0, 10.0, i0=1.0),
                ProbePosition(2, 2.0, 30.0, i0=1.0),
                ProbePosition(2, 4.0, 50.0, i0=1.0),
            ]
        )

        in_range, paired = seq.interpolate_at_indexes(numpy.array([5, 2, 1, -1]))

        numpy.testing.assert_array_equal(in_range, [False, True, True, False])
        numpy.testing.assert_array_equal(paired.get_indexes(), [2, 1])
        numpy.testing.assert_allclose(paired.get_coordinates_x_m(), [3.0, 1.5])
        numpy.testing.assert_allclose(paired.get_coordinates_y_m(), [40.0, 25.0])
        assert paired.get_i0() is None


class TestMatchIndexes:
    def test_pairs_only_shared_indexes_and_averages_duplicates(self) -> None:
        reference = ProbePositionSequence(
            [
                ProbePosition(index=0, x_m=0.0, y_m=0.0),
                ProbePosition(index=1, x_m=1.0, y_m=10.0),
                ProbePosition(index=1, x_m=3.0, y_m=30.0),
                ProbePosition(index=2, x_m=2.0, y_m=20.0),
                ProbePosition(index=4, x_m=4.0, y_m=40.0),
            ]
        )
        test = ProbePositionSequence(
            [
                ProbePosition(index=4, x_m=4.5, y_m=40.5),
                ProbePosition(index=1, x_m=2.5, y_m=20.5),
                ProbePosition(index=3, x_m=3.5, y_m=30.5),
            ]
        )

        indexes, reference_xy_m, test_xy_m = reference.match_indexes(test)

        numpy.testing.assert_array_equal(indexes, [1, 4])
        numpy.testing.assert_allclose(reference_xy_m, [[2.0, 20.0], [4.0, 40.0]])
        numpy.testing.assert_allclose(test_xy_m, [[2.5, 20.5], [4.5, 40.5]])

    def test_index_missing_from_one_side_is_not_interpolated(self) -> None:
        reference = ProbePositionSequence(
            [ProbePosition(index=i, x_m=float(i), y_m=0.0) for i in (0, 1, 2)]
        )
        test = ProbePositionSequence(
            [ProbePosition(index=i, x_m=float(i), y_m=0.0) for i in (0, 2)]
        )

        indexes, reference_xy_m, test_xy_m = reference.match_indexes(test)

        numpy.testing.assert_array_equal(indexes, [0, 2])
        assert reference_xy_m.shape == test_xy_m.shape == (2, 2)

    def test_disjoint_sequences_share_nothing(self) -> None:
        reference = ProbePositionSequence([ProbePosition(index=0, x_m=0.0, y_m=0.0)])
        test = ProbePositionSequence([ProbePosition(index=1, x_m=0.0, y_m=0.0)])

        indexes, reference_xy_m, test_xy_m = reference.match_indexes(test)

        assert indexes.size == 0
        assert reference_xy_m.shape == test_xy_m.shape == (0, 2)
