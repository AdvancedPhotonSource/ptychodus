"""Unit tests for ptychodus.api.preprocess.diffraction.estimate_crop_center.

Behaviors verified for Fresnel-zone-plate (with central stop) patterns:
  - Centered annular pattern returns the geometric center
  - Off-center pattern returns the correct (y, x) pixel
  - Single saturated hot pixel does not bias the result
  - Bad-pixel mask suppresses arbitrarily large outliers
  - Heavy Gaussian background noise does not pull the centroid toward
    the detector center
  - Integer-dtype input is accepted
  - Combined corruption (noise + masked hot pixels + unmasked hot pixel) still
    produces the correct center
  - All-bad input falls back to the geometric center
  - A bright asymmetric off-center feature does not bias the centroid (two-pass)
"""

import numpy
import pytest

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.diffraction import (
    CropCenter,
    DiffractionMetadata,
    SimpleDiffractionArray,
)
from ptychodus.api.preprocess.diffraction import estimate_crop_center
from ptychodus.api.geometry import PixelGeometry


def _fzp_pattern(
    shape: tuple[int, int],
    center_yx: tuple[float, float],
    inner_radius: float = 8.0,
    outer_radius: float = 14.0,
    peak_intensity: float = 1000.0,
) -> numpy.ndarray:
    """Synthetic Fresnel-zone-plate-with-central-stop intensity: bright annulus, dark core."""
    height, width = shape
    y = numpy.arange(height, dtype=numpy.float64).reshape(-1, 1) - center_yx[0]
    x = numpy.arange(width, dtype=numpy.float64).reshape(1, -1) - center_yx[1]
    r = numpy.hypot(y, x)
    midline = 0.5 * (inner_radius + outer_radius)
    annulus = peak_intensity * numpy.exp(-((r - midline) ** 2) / 2.0)
    annulus[r < inner_radius] = 0.0
    return annulus


def test_centered_pattern_returns_geometric_center() -> None:
    """A centrosymmetric FZP pattern centered on the detector returns the center."""
    pattern = _fzp_pattern((64, 64), (31.5, 31.5))
    assert estimate_crop_center(pattern) == CropCenter(position_x_px=32, position_y_px=32)


def test_offset_pattern_returns_offset_center() -> None:
    """An off-center FZP pattern returns the correct pixel."""
    pattern = _fzp_pattern((64, 80), (20.0, 50.0))
    assert estimate_crop_center(pattern) == CropCenter(position_x_px=50, position_y_px=20)


def test_unmasked_hot_pixel_does_not_bias_centroid() -> None:
    """A single saturated hot pixel far from the pattern is suppressed by median filtering."""
    pattern = _fzp_pattern((64, 64), (31.5, 31.5))
    pattern[5, 5] = 1.0e6
    center = estimate_crop_center(pattern)
    assert abs(center.position_y_px - 32) <= 1
    assert abs(center.position_x_px - 32) <= 1


def test_bad_pixel_mask_excludes_extreme_outliers() -> None:
    """Hot pixels flagged in bad_pixels are fully ignored even at extreme magnitudes."""
    pattern = _fzp_pattern((64, 64), (31.5, 31.5))
    bad = numpy.zeros((64, 64), dtype=bool)
    pattern[2, 2] = 1.0e9
    pattern[60, 60] = 5.0e8
    bad[2, 2] = True
    bad[60, 60] = True
    assert estimate_crop_center(pattern, bad) == CropCenter(position_x_px=32, position_y_px=32)


def test_robust_to_gaussian_background_noise() -> None:
    """Background noise across the full detector does not pull the centroid to its center."""
    rng = numpy.random.default_rng(42)
    pattern = _fzp_pattern((64, 64), (20.0, 45.0))
    pattern = pattern + rng.normal(0.0, 30.0, size=pattern.shape)
    numpy.clip(pattern, 0.0, None, out=pattern)
    center = estimate_crop_center(pattern)
    assert abs(center.position_y_px - 20) <= 1
    assert abs(center.position_x_px - 45) <= 1


def test_accepts_integer_dtype() -> None:
    """Integer (Poisson-count-like) input is handled correctly."""
    pattern = _fzp_pattern((48, 48), (23.5, 23.5)).astype(numpy.uint16)
    assert estimate_crop_center(pattern) == CropCenter(position_x_px=24, position_y_px=24)


def test_combined_corruption_still_recovers_center() -> None:
    """Noise + masked hot pixels + an unmasked transient still yields the correct center."""
    rng = numpy.random.default_rng(7)
    pattern = _fzp_pattern((96, 96), (40.0, 55.0))
    pattern = pattern + rng.normal(0.0, 20.0, size=pattern.shape)
    numpy.clip(pattern, 0.0, None, out=pattern)
    bad = numpy.zeros((96, 96), dtype=bool)
    for yx in [(3, 3), (90, 4), (7, 88)]:
        pattern[yx] = 1.0e7
        bad[yx] = True
    # One stray hot pixel left unflagged - must be killed by the median filter.
    pattern[85, 85] = 5.0e5
    center = estimate_crop_center(pattern, bad)
    assert abs(center.position_y_px - 40) <= 1
    assert abs(center.position_x_px - 55) <= 1


def test_all_bad_falls_back_to_geometric_center() -> None:
    """When every pixel is masked, the function returns the geometric center."""
    pattern = numpy.ones((10, 12), dtype=numpy.uint16)
    bad = numpy.ones((10, 12), dtype=bool)
    assert estimate_crop_center(pattern, bad) == CropCenter(position_x_px=6, position_y_px=5)


def test_all_zero_input_falls_back_to_geometric_center() -> None:
    """A blank detector has no signal and must not divide by zero."""
    pattern = numpy.zeros((20, 20), dtype=numpy.float64)
    assert estimate_crop_center(pattern) == CropCenter(position_x_px=10, position_y_px=10)


@pytest.mark.parametrize('center_yx', [(15.0, 15.0), (10.0, 25.0), (28.0, 12.0)])
def test_no_bad_pixels_argument_matches_explicit_none(
    center_yx: tuple[float, float],
) -> None:
    """Passing None for bad_pixels is equivalent to omitting it."""
    pattern = _fzp_pattern((40, 40), center_yx)
    assert estimate_crop_center(pattern) == estimate_crop_center(pattern, None)


def test_robust_across_noise_threshold_choice() -> None:
    """Both strict and loose noise_threshold values still recover the centroid.

    Even when pass 1 is biased by an un-rejected noise floor, pass 2 narrows
    around the pass-1 estimate and recentroids on the signal in the window.
    """
    rng = numpy.random.default_rng(123)
    pattern = _fzp_pattern((64, 64), (12.0, 12.0))
    pattern = pattern + rng.normal(0.0, 25.0, size=pattern.shape)
    numpy.clip(pattern, 0.0, None, out=pattern)

    for threshold in (5.0, 0.0):
        center = estimate_crop_center(pattern, mad_threshold=threshold)
        assert abs(center.position_y_px - 12) <= 1
        assert abs(center.position_x_px - 12) <= 1


def test_asymmetric_bright_peak_does_not_bias_centroid() -> None:
    """A bright off-center feature pulls pass 1 only slightly; pass 2 excludes it."""
    pattern = _fzp_pattern((64, 64), (31.5, 31.5))
    y = numpy.arange(64, dtype=numpy.float64).reshape(-1, 1)
    x = numpy.arange(64, dtype=numpy.float64).reshape(1, -1)
    blob = 300.0 * numpy.exp(-(((y - 8.0) ** 2 + (x - 8.0) ** 2) / (2.0 * 3.0**2)))
    center = estimate_crop_center(pattern + blob)
    assert abs(center.position_y_px - 32) <= 1
    assert abs(center.position_x_px - 32) <= 1


class TestNbytes:
    def test_simple_array_sums_indexes_and_patterns(self) -> None:
        indexes = numpy.arange(3, dtype=numpy.int32)
        patterns = numpy.zeros((3, 4, 5), dtype=numpy.uint16)
        array = SimpleDiffractionArray('a', indexes, patterns)

        assert array.nbytes == indexes.nbytes + patterns.nbytes

    def test_assembled_data_sums_all_three_arrays(self) -> None:
        indexes = numpy.arange(3, dtype=numpy.int32)
        patterns = numpy.zeros((3, 4, 5), dtype=numpy.uint16)
        bad_pixels = numpy.zeros((4, 5), dtype=numpy.bool_)
        data = AssembledDiffractionData(
            indexes, patterns, PixelGeometry(width_m=1e-4, height_m=1e-4), bad_pixels
        )

        assert data.nbytes == indexes.nbytes + patterns.nbytes + bad_pixels.nbytes

    def test_metadata_nbytes_is_positive(self) -> None:
        metadata = DiffractionMetadata.create_null()

        assert metadata.nbytes > 0
