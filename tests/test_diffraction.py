"""Unit tests for ptychodus.api.preprocess.diffraction.estimate_beam_center.

Behaviors verified for Fresnel-zone-plate (with central stop) patterns:
  - Centered annular pattern returns the geometric center
  - Off-center pattern returns the correct (y, x) pixel
  - Single saturated hot pixel does not bias the result
  - Heavy Gaussian background noise does not pull the centroid toward
    the detector center
  - Integer-dtype input is accepted
  - Combined corruption (noise + inpainted bad pixels + unmasked hot pixel)
    still produces the correct center
  - A bright asymmetric off-center feature does not bias the centroid (two-pass)
"""

import numpy

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.diffraction import (
    BeamCenter,
    CropRegion,
    DiffractionMetadata,
    SimpleDiffractionArray,
)
from ptychodus.api.preprocess.diffraction import estimate_beam_center, inpaint_bad_pixels
from ptychodus.api.geometry import ImageExtent, PixelGeometry


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
    assert estimate_beam_center(pattern) == BeamCenter(x_px=32, y_px=32)


def test_offset_pattern_returns_offset_center() -> None:
    """An off-center FZP pattern returns the correct pixel."""
    pattern = _fzp_pattern((64, 80), (20.0, 50.0))
    assert estimate_beam_center(pattern) == BeamCenter(x_px=50, y_px=20)


def test_unmasked_hot_pixel_does_not_bias_centroid() -> None:
    """A single saturated hot pixel far from the pattern is suppressed by median filtering."""
    pattern = _fzp_pattern((64, 64), (31.5, 31.5))
    pattern[5, 5] = 1.0e6
    center = estimate_beam_center(pattern)
    assert abs(center.y_px - 32) <= 1
    assert abs(center.x_px - 32) <= 1


def test_robust_to_gaussian_background_noise() -> None:
    """Background noise across the full detector does not pull the centroid to its center."""
    rng = numpy.random.default_rng(42)
    pattern = _fzp_pattern((64, 64), (20.0, 45.0))
    pattern = pattern + rng.normal(0.0, 30.0, size=pattern.shape)
    numpy.clip(pattern, 0.0, None, out=pattern)
    center = estimate_beam_center(pattern)
    assert abs(center.y_px - 20) <= 1
    assert abs(center.x_px - 45) <= 1


def test_accepts_integer_dtype() -> None:
    """Integer (Poisson-count-like) input is handled correctly."""
    pattern = _fzp_pattern((48, 48), (23.5, 23.5)).astype(numpy.uint16)
    assert estimate_beam_center(pattern) == BeamCenter(x_px=24, y_px=24)


def test_combined_corruption_still_recovers_center() -> None:
    """Noise + inpainted bad pixels + an unmasked transient still yields the correct center."""
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
    center = estimate_beam_center(inpaint_bad_pixels(pattern, bad))
    assert abs(center.y_px - 40) <= 1
    assert abs(center.x_px - 55) <= 1


def test_all_zero_input_falls_back_to_geometric_center() -> None:
    """A blank detector has no signal and must not divide by zero."""
    pattern = numpy.zeros((20, 20), dtype=numpy.float64)
    assert estimate_beam_center(pattern) == BeamCenter(x_px=10, y_px=10)


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
        center = estimate_beam_center(pattern, mad_threshold=threshold)
        assert abs(center.y_px - 12) <= 1
        assert abs(center.x_px - 12) <= 1


def test_asymmetric_bright_peak_does_not_bias_centroid() -> None:
    """A bright off-center feature pulls pass 1 only slightly; pass 2 excludes it."""
    pattern = _fzp_pattern((64, 64), (31.5, 31.5))
    y = numpy.arange(64, dtype=numpy.float64).reshape(-1, 1)
    x = numpy.arange(64, dtype=numpy.float64).reshape(1, -1)
    blob = 300.0 * numpy.exp(-(((y - 8.0) ** 2 + (x - 8.0) ** 2) / (2.0 * 3.0**2)))
    center = estimate_beam_center(pattern + blob)
    assert abs(center.y_px - 32) <= 1
    assert abs(center.x_px - 32) <= 1


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


class TestCropRegionFromCenterExtent:
    def test_combines_fields(self) -> None:
        region = CropRegion.from_center_extent(
            BeamCenter(x_px=10, y_px=20), ImageExtent(width_px=4, height_px=6)
        )
        # center 10, width 4 → x_start = 10 - 2 = 8, x_range = [8, 12)
        # center 20, height 6 → y_start = 20 - 3 = 17, y_range = [17, 23)
        assert region == CropRegion(x_range=(8, 12), y_range=(17, 23))
        assert (region.width_px, region.height_px) == (4, 6)
        assert (region.center_x_px, region.center_y_px) == (10, 20)

    def test_odd_width_preserves_requested_size(self) -> None:
        # Odd width used to silently narrow by one under the center + width // 2 math.
        region = CropRegion.from_center_extent(
            BeamCenter(x_px=10, y_px=10), ImageExtent(width_px=5, height_px=5)
        )
        assert (region.width_px, region.height_px) == (5, 5)
        # Bias one pixel toward the origin: start = 10 - 5 // 2 = 8.
        assert region.x_range == (8, 13)
        assert region.y_range == (8, 13)


class TestCropRegionFromLargestPow2:
    def test_centered_gives_largest_pow2_that_fits(self) -> None:
        # detector 64x64, center (32,32): max_radius = 32 → size = 64 (2 * 32).
        region = CropRegion.from_largest_pow2(
            BeamCenter(x_px=32, y_px=32), ImageExtent(width_px=64, height_px=64)
        )
        assert (region.width_px, region.height_px) == (64, 64)

    def test_offset_center_rounds_down_to_pow2(self) -> None:
        # detector 64x64, center (10, 32): max_radius = min(10, 54, 32, 32) = 10 → 2 * 10 = 20
        # largest pow2 <= 20 is 16.
        region = CropRegion.from_largest_pow2(
            BeamCenter(x_px=10, y_px=32), ImageExtent(width_px=64, height_px=64)
        )
        assert (region.width_px, region.height_px) == (16, 16)
        assert (region.center_x_px, region.center_y_px) == (10, 32)

    def test_center_on_boundary_returns_zero_size(self) -> None:
        region = CropRegion.from_largest_pow2(
            BeamCenter(x_px=0, y_px=32), ImageExtent(width_px=64, height_px=64)
        )
        assert (region.width_px, region.height_px) == (0, 0)

    def test_asymmetric_detector_uses_tightest_axis(self) -> None:
        # center (32, 32), detector 128x40: max_radius = min(32, 96, 32, 8) = 8 → 2 * 8 = 16.
        region = CropRegion.from_largest_pow2(
            BeamCenter(x_px=32, y_px=32), ImageExtent(width_px=128, height_px=40)
        )
        assert (region.width_px, region.height_px) == (16, 16)


class TestCropRegionClampToDetectorExtent:
    def test_in_bounds_region_is_unchanged(self) -> None:
        region = CropRegion(x_range=(2, 6), y_range=(2, 6))
        assert region.clamp_to_detector_extent(ImageExtent(width_px=8, height_px=8)) == region

    def test_range_shifted_to_fit(self) -> None:
        # center 0, width 4 → x_range = [-2, 2); must shift so start >= 0.
        region = CropRegion.from_center_extent(
            BeamCenter(x_px=0, y_px=4), ImageExtent(width_px=4, height_px=4)
        )
        clamped = region.clamp_to_detector_extent(ImageExtent(width_px=8, height_px=8))
        assert clamped == CropRegion(x_range=(0, 4), y_range=(2, 6))

    def test_extent_shrunk_to_detector(self) -> None:
        # requested 20x20 in an 8x8 detector: cap at 8x8, then range clamped to [0, 8).
        region = CropRegion.from_center_extent(
            BeamCenter(x_px=100, y_px=100), ImageExtent(width_px=20, height_px=20)
        )
        clamped = region.clamp_to_detector_extent(ImageExtent(width_px=8, height_px=8))
        assert clamped == CropRegion(x_range=(0, 8), y_range=(0, 8))

    def test_zero_width_floors_to_one(self) -> None:
        region = CropRegion(x_range=(4, 4), y_range=(4, 4))
        clamped = region.clamp_to_detector_extent(ImageExtent(width_px=8, height_px=8))
        assert (clamped.width_px, clamped.height_px) == (1, 1)
