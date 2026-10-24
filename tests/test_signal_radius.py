"""Unit tests for ptychodus.api.preprocess.diffraction.estimate_signal_radius.

The synthetic patterns are a Gaussian decay over a Poisson background, whose crossing of
the threshold the estimator reports is known in closed form:
``peak * exp(-r**2 / (2 * sigma**2)) == threshold``.
"""

import numpy
import pytest

from ptychodus.api.diffraction import BeamCenter
from ptychodus.api.preprocess.diffraction import estimate_signal_radius

_SHAPE = (256, 256)
_PEAK = 1000.0
_BACKGROUND = 5.0
_SIGMA = 10.0


def _gaussian_pattern(center: BeamCenter, *, sigma: float = _SIGMA, seed: int = 0) -> numpy.ndarray:
    height, width = _SHAPE
    y = numpy.arange(height, dtype=numpy.float64).reshape(-1, 1) - center.y_px
    x = numpy.arange(width, dtype=numpy.float64).reshape(1, -1) - center.x_px
    expected = _PEAK * numpy.exp(-(x**2 + y**2) / (2.0 * sigma**2)) + _BACKGROUND
    return numpy.random.default_rng(seed).poisson(expected).astype(numpy.float64)


def _crossing_radius(threshold: float, *, sigma: float = _SIGMA) -> float:
    return sigma * numpy.sqrt(2.0 * numpy.log(_PEAK / threshold))


def test_radius_is_where_the_decay_meets_the_threshold() -> None:
    center = BeamCenter(x_px=128, y_px=128)
    signal = estimate_signal_radius(_gaussian_pattern(center), center)

    assert signal.noise_floor.median == pytest.approx(_BACKGROUND, abs=1.0)
    assert signal.threshold > 0.0
    assert abs(signal.radius_px - _crossing_radius(signal.threshold)) <= 2.0


def test_integer_patterns_are_accepted() -> None:
    center = BeamCenter(x_px=128, y_px=128)
    pattern = _gaussian_pattern(center)
    float_signal = estimate_signal_radius(pattern, center)
    integer_signal = estimate_signal_radius(pattern.astype(numpy.uint16), center)

    assert integer_signal.radius_px == float_signal.radius_px


def test_a_beamstop_shadow_does_not_end_the_search() -> None:
    center = BeamCenter(x_px=128, y_px=128)
    pattern = _gaussian_pattern(center)
    clear_radius = estimate_signal_radius(pattern, center).radius_px

    y, x = numpy.ogrid[: _SHAPE[0], : _SHAPE[1]]
    pattern[numpy.hypot(x - center.x_px, y - center.y_px) < 8.0] = 0.0

    assert estimate_signal_radius(pattern, center).radius_px == clear_radius


def test_an_off_center_beam_is_judged_on_partial_rings() -> None:
    centered = BeamCenter(x_px=128, y_px=128)
    off_center = BeamCenter(x_px=40, y_px=200)
    centered_signal = estimate_signal_radius(_gaussian_pattern(centered), centered)
    signal = estimate_signal_radius(_gaussian_pattern(off_center), off_center)

    assert abs(signal.radius_px - centered_signal.radius_px) <= 1
    # Rings out to the farthest corner are kept, and the last of them are too sparse to judge.
    farthest_corner = numpy.hypot(_SHAPE[1] - 1 - off_center.x_px, off_center.y_px)
    assert signal.radial_profile.size == int(farthest_corner) + 1
    assert numpy.isnan(signal.radial_profile[-1])


def test_masked_pixels_are_left_out_of_every_ring() -> None:
    center = BeamCenter(x_px=128, y_px=128)
    pattern = _gaussian_pattern(center)
    clear_radius = estimate_signal_radius(pattern, center).radius_px

    # A hot stripe running off the detector edge, wide enough to survive the median filter.
    bad_pixels = numpy.zeros(_SHAPE, dtype=bool)
    bad_pixels[:, 230:236] = True
    pattern[bad_pixels] = 1.0e6

    assert estimate_signal_radius(pattern, center).radius_px > clear_radius + 50
    assert estimate_signal_radius(pattern, center, bad_pixels=bad_pixels).radius_px == clear_radius


def test_pure_noise_has_no_signal_radius() -> None:
    pattern = numpy.random.default_rng(1).poisson(_BACKGROUND, size=_SHAPE).astype(numpy.float64)
    signal = estimate_signal_radius(pattern, BeamCenter(x_px=128, y_px=128))

    assert signal.radius_px == 0


def test_a_zero_noise_floor_counts_any_ring_above_background() -> None:
    """A mostly-empty photon-counting frame: median and MAD are both zero."""
    center = BeamCenter(x_px=128, y_px=128)
    y, x = numpy.ogrid[: _SHAPE[0], : _SHAPE[1]]
    pattern = numpy.zeros(_SHAPE, dtype=numpy.uint32)
    pattern[numpy.hypot(x - center.x_px, y - center.y_px) <= 20.0] = 3

    signal = estimate_signal_radius(pattern, center)

    assert signal.noise_floor.median_absolute_deviation == 0.0
    assert signal.threshold == 0.0
    assert abs(signal.radius_px - 20) <= 1


def test_a_larger_mad_threshold_reports_a_smaller_radius() -> None:
    center = BeamCenter(x_px=128, y_px=128)
    pattern = _gaussian_pattern(center)

    loose = estimate_signal_radius(pattern, center, mad_threshold=2.0)
    strict = estimate_signal_radius(pattern, center, mad_threshold=20.0)

    assert strict.radius_px < loose.radius_px


def test_an_all_bad_mask_is_an_error() -> None:
    center = BeamCenter(x_px=128, y_px=128)

    with pytest.raises(ValueError, match='Every pixel is bad'):
        estimate_signal_radius(
            _gaussian_pattern(center), center, bad_pixels=numpy.ones(_SHAPE, dtype=bool)
        )


def test_a_negative_mad_threshold_is_an_error() -> None:
    center = BeamCenter(x_px=128, y_px=128)

    with pytest.raises(ValueError, match='non-negative'):
        estimate_signal_radius(_gaussian_pattern(center), center, mad_threshold=-1.0)
