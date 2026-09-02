"""Unit tests for the optional per-position photon-count field on ProbePosition.

ProbePositionSequence enforces an all-or-nothing invariant: every point supplies a
probe_photon_count or none do. A mix indicates a reader bug and is rejected at
construction with ValueError.
"""

from __future__ import annotations

import numpy
import pytest

from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence


class TestProbePositionPhotonCount:
    def test_default_is_none(self) -> None:
        point = ProbePosition(index=0, x_m=1.0, y_m=2.0)
        assert point.probe_photon_count is None

    def test_value_is_carried(self) -> None:
        point = ProbePosition(
            index=0, x_m=1.0, y_m=2.0, probe_photon_count=7.5
        )
        assert point.probe_photon_count == pytest.approx(7.5)


class TestProbePositionSequencePhotonCounts:
    def test_absent_when_no_point_supplies_one(self) -> None:
        seq = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(3)])
        assert seq.get_probe_photon_counts() is None

    def test_present_when_every_point_supplies_one(self) -> None:
        seq = ProbePositionSequence(
            [
                ProbePosition(0, 0.0, 0.0, probe_photon_count=10.0),
                ProbePosition(1, 1.0, 0.0, probe_photon_count=20.0),
                ProbePosition(2, 2.0, 0.0, probe_photon_count=30.0),
            ]
        )
        counts = seq.get_probe_photon_counts()
        assert counts is not None
        assert counts.shape == (3,)
        numpy.testing.assert_array_equal(counts, [10.0, 20.0, 30.0])

    def test_mixed_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='every point or none'):
            ProbePositionSequence(
                [
                    ProbePosition(0, 0.0, 0.0, probe_photon_count=10.0),
                    ProbePosition(1, 1.0, 0.0),
                    ProbePosition(2, 2.0, 0.0, probe_photon_count=30.0),
                ]
            )

    def test_negative_probe_photon_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='finite non-negative'):
            ProbePositionSequence(
                [
                    ProbePosition(0, 0.0, 0.0, probe_photon_count=10.0),
                    ProbePosition(1, 1.0, 0.0, probe_photon_count=-1.0),
                    ProbePosition(2, 2.0, 0.0, probe_photon_count=30.0),
                ]
            )

    def test_non_finite_probe_photon_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='finite non-negative'):
            ProbePositionSequence(
                [
                    ProbePosition(0, 0.0, 0.0, probe_photon_count=10.0),
                    ProbePosition(1, 1.0, 0.0, probe_photon_count=float('nan')),
                    ProbePosition(2, 2.0, 0.0, probe_photon_count=30.0),
                ]
            )

    def test_iteration_yields_populated_counts(self) -> None:
        seq = ProbePositionSequence(
            [
                ProbePosition(0, 0.0, 0.0, probe_photon_count=10.0),
                ProbePosition(1, 1.0, 0.0, probe_photon_count=20.0),
            ]
        )
        points = list(seq)
        assert points[0].probe_photon_count == pytest.approx(10.0)
        assert points[1].probe_photon_count == pytest.approx(20.0)

    def test_iteration_yields_none_when_no_counts(self) -> None:
        seq = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(2)])
        for point in seq:
            assert point.probe_photon_count is None

    def test_slice_preserves_counts(self) -> None:
        seq = ProbePositionSequence(
            [ProbePosition(i, float(i), 0.0, probe_photon_count=float(10 + i)) for i in range(5)]
        )
        subset = seq[1:4]
        assert len(subset) == 3
        as_list = list(subset)
        assert as_list[0].index == 1
        assert as_list[0].probe_photon_count == pytest.approx(11.0)
        assert as_list[2].probe_photon_count == pytest.approx(13.0)

    def test_slice_stays_uniform_when_no_counts_supplied(self) -> None:
        seq = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(5)])
        subset = seq[1:4]
        assert subset.get_probe_photon_counts() is None

    def test_copy_preserves_counts_and_is_independent(self) -> None:
        seq = ProbePositionSequence(
            [
                ProbePosition(0, 0.0, 0.0, probe_photon_count=10.0),
                ProbePosition(1, 1.0, 0.0, probe_photon_count=20.0),
            ]
        )
        clone = seq.copy()
        original = seq.get_probe_photon_counts()
        cloned = clone.get_probe_photon_counts()
        assert original is not None
        assert cloned is not None
        assert cloned is not original
        numpy.testing.assert_array_equal(cloned, original)

    def test_nbytes_includes_counts(self) -> None:
        without = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(3)])
        with_counts = ProbePositionSequence(
            [ProbePosition(i, float(i), 0.0, probe_photon_count=float(i)) for i in range(3)]
        )
        assert with_counts.nbytes > without.nbytes
