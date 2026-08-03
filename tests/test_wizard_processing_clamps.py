"""Unit tests for the wizard-side clamp helpers.

These lock the pure functions used by `CropViewController` and
`BinningViewController` to compute spin-box ranges.
"""

from ptychodus.api.geometry import Interval
from ptychodus.controller.diffraction.wizard.processing import (
    _bin_size_limits,
    _crop_center_limits,
    _crop_size_limits,
    _effective_crop_size,
)


def _bounds(interval: Interval[int]) -> tuple[int, int]:
    """`Interval` lacks value equality; compare (lower, upper) tuples instead."""
    return interval.lower, interval.upper


def test_crop_size_limits_lower_is_one() -> None:
    assert _bounds(_crop_size_limits(64)) == (1, 64)


def test_crop_center_limits_lower_is_one() -> None:
    assert _bounds(_crop_center_limits(32)) == (1, 32)


def test_effective_crop_size_uses_requested_when_enabled() -> None:
    assert _effective_crop_size(64, 32, crop_enabled=True) == 32


def test_effective_crop_size_clamps_request_to_detector() -> None:
    assert _effective_crop_size(64, 999, crop_enabled=True) == 64
    assert _effective_crop_size(64, 0, crop_enabled=True) == 1


def test_effective_crop_size_ignores_request_when_disabled() -> None:
    assert _effective_crop_size(64, 32, crop_enabled=False) == 64


def test_bin_size_limits_upper_is_effective_crop() -> None:
    assert _bounds(_bin_size_limits(32)) == (1, 32)
