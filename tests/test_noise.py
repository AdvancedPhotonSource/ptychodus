"""Unit tests for ptychodus.api.preprocess.noise.

Behaviors verified:
  - compute_robust_statistics returns median and MAD for odd- and even-length inputs
  - RobustStatistics.get_bounds returns the symmetric interval [median - k*MAD, median + k*MAD]
  - require_positive clips the lower bound to nextafter(0, 1) when the raw lower is <= 0
  - Zero-MAD degenerate cases return unbounded intervals on the undefined side
  - Interval width scales linearly with k
  - get_bounds(k).upper matches get_significance_threshold(k)
  - Interval.__contains__ correctly separates outliers from the bulk
"""

import numpy
import pytest

from ptychodus.api.geometry import Interval
from ptychodus.api.preprocess.noise import (
    RobustStatistics,
    compute_robust_statistics,
)


def test_compute_robust_statistics_odd_length() -> None:
    stats = compute_robust_statistics(numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert stats.median == 3.0
    assert stats.median_absolute_deviation == 1.0


def test_compute_robust_statistics_even_length_averages_middles() -> None:
    stats = compute_robust_statistics(numpy.array([1.0, 2.0, 3.0, 4.0]))
    # median of [1,2,3,4] = (2+3)/2 = 2.5; abs deviations = [1.5, 0.5, 0.5, 1.5];
    # median of sorted [0.5, 0.5, 1.5, 1.5] = (0.5+1.5)/2 = 1.0.
    assert stats.median == 2.5
    assert stats.median_absolute_deviation == 1.0


def test_get_bounds_symmetric_distribution() -> None:
    stats = compute_robust_statistics(numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    interval = stats.get_bounds(k=2.0)
    assert interval.lower == 1.0  # 3 - 2*1
    assert interval.upper == 5.0  # 3 + 2*1


def test_get_bounds_require_positive_clips_negative_lower_to_nextafter_zero() -> None:
    stats = compute_robust_statistics(numpy.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    # median = 0.3, MAD = 0.1; k=5 gives raw lower = -0.2, which is <= 0.
    interval = stats.get_bounds(k=5.0, require_positive=True)
    assert interval.lower == float(numpy.nextafter(0.0, 1.0))
    assert interval.upper == pytest.approx(0.8)


def test_get_bounds_default_does_not_require_positive() -> None:
    stats = compute_robust_statistics(numpy.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    interval = stats.get_bounds(k=5.0)  # default require_positive=False
    # Raw lower is negative; without require_positive it is returned unchanged.
    assert interval.lower == pytest.approx(-0.2)


def test_get_bounds_zero_mad_without_require_positive_is_unbounded() -> None:
    stats = RobustStatistics(median=7.0, median_absolute_deviation=0.0)
    interval = stats.get_bounds(k=3.0)
    assert interval.lower == -numpy.inf
    assert interval.upper == numpy.inf


def test_get_bounds_zero_mad_with_require_positive_is_lower_bounded() -> None:
    stats = RobustStatistics(median=7.0, median_absolute_deviation=0.0)
    interval = stats.get_bounds(k=3.0, require_positive=True)
    assert interval.lower == float(numpy.nextafter(0.0, 1.0))
    assert interval.upper == numpy.inf


def test_get_bounds_width_scales_linearly_with_k() -> None:
    stats = compute_robust_statistics(numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    narrow = stats.get_bounds(k=1.0)
    wide = stats.get_bounds(k=2.0)
    assert (wide.upper - wide.lower) == 2.0 * (narrow.upper - narrow.lower)


def test_get_bounds_upper_matches_get_significance_threshold() -> None:
    stats = compute_robust_statistics(numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    for k in (0.5, 1.0, 2.5, 4.5):
        assert stats.get_bounds(k=k).upper == stats.get_significance_threshold(k)


def test_get_bounds_rejects_outlier_while_keeping_bulk() -> None:
    # median = 11, MAD = 1; k=3 gives [8, 14].
    stats = compute_robust_statistics(numpy.array([10.0, 11.0, 10.0, 12.0, 500.0]))
    interval = stats.get_bounds(k=3.0)
    assert 500.0 not in interval
    assert 10.0 in interval
    assert 11.0 in interval
    assert 12.0 in interval


def test_get_bounds_returns_interval_of_float() -> None:
    stats = compute_robust_statistics(numpy.array([1.0, 2.0, 3.0]))
    assert isinstance(stats.get_bounds(k=2.0), Interval)
