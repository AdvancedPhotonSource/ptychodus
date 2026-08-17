import numpy
import pytest

from ptychodus.api.interpolate import (
    BarycentricArrayInterpolator,
    BarycentricArrayStitcher,
    lerp,
)


def test_lerp_scalar_endpoints() -> None:
    assert lerp(0.0, 10.0, 0.0) == pytest.approx(0.0)
    assert lerp(0.0, 10.0, 1.0) == pytest.approx(10.0)
    assert lerp(0.0, 10.0, 0.25) == pytest.approx(2.5)


def test_barycentric_get_patch_matches_analytic_bilinear_on_plane() -> None:
    """For a plane f(x, y) = 2*x + 3*y + 1, bilinear interpolation is exact."""
    ny, nx = 12, 14
    ys, xs = numpy.mgrid[0:ny, 0:nx].astype(float)
    array = 2.0 * xs + 3.0 * ys + 1.0

    interp = BarycentricArrayInterpolator(array)
    patch = interp.get_patch(center_x=6.3, center_y=5.7, width=4, height=4)

    # The patch samples correspond to integer pixel positions relative to the
    # top-left of the (whole) support slice. Reconstruct the expected values
    # analytically by knowing where each pixel lands in world coordinates.
    x_lower = 6.3 - 2.0  # center_x - width/2
    y_lower = 5.7 - 2.0
    # The gather is over an (n+1)x(n+1) support and produces an nxn patch
    # whose samples are at whole indices x_lower_wh + xmin_fr + i for i in 0..n-1.
    # For a plane, sampling at fractional (x_lower + i, y_lower + j) is exact.
    px, py = numpy.meshgrid(x_lower + numpy.arange(4), y_lower + numpy.arange(4), indexing='xy')
    expected = 2.0 * px + 3.0 * py + 1.0
    numpy.testing.assert_allclose(patch, expected, atol=1e-12)


def test_barycentric_add_patch_scatters_full_intensity_at_integer_center() -> None:
    """When the sub-pixel offset is zero, all patch mass lands on the top-left corner weight."""
    array = numpy.zeros((10, 10), dtype=float)
    interp = BarycentricArrayInterpolator(array)

    patch = numpy.ones((3, 3), dtype=float)
    # width/2 = 1.5; pick center so that x_lower and y_lower are exact half-integers,
    # yielding x_frac = y_frac = 0.5 (weights all 0.25).
    interp.add_patch(center_x=4.5, center_y=4.5, patch=patch)

    # Only 4 corners of the 4x4 support carry weight 0.25 * 1.0 = 0.25 per patch pixel;
    # interior pixels see contributions from all 4 corners for a total of 1.0.
    # Sum over the affected region must equal sum(patch) = 9.0.
    assert array.sum() == pytest.approx(patch.sum())


def test_barycentric_add_patch_is_transpose_of_get_patch() -> None:
    """<get_patch(a), b> == <a, add_patch(0, b)> for all a, b (adjoint identity)."""
    rng = numpy.random.default_rng(2026)
    ny, nx = 16, 20
    a = rng.standard_normal((ny, nx))
    b = rng.standard_normal((5, 6))

    center_x = 9.3
    center_y = 7.8

    forward = BarycentricArrayInterpolator(a)
    patch = forward.get_patch(center_x, center_y, b.shape[-1], b.shape[-2])
    lhs = float(numpy.sum(patch * b))

    scatter_target = numpy.zeros_like(a)
    adjoint = BarycentricArrayInterpolator(scatter_target)
    adjoint.add_patch(center_x, center_y, b)
    rhs = float(numpy.sum(a * scatter_target))

    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_barycentric_add_patch_accumulates() -> None:
    """Two calls at the same center add — no state reset between calls."""
    array = numpy.zeros((10, 10), dtype=float)
    interp = BarycentricArrayInterpolator(array)

    patch = numpy.ones((3, 3), dtype=float)
    interp.add_patch(center_x=4.5, center_y=4.5, patch=patch)
    interp.add_patch(center_x=4.5, center_y=4.5, patch=patch)

    assert array.sum() == pytest.approx(2 * patch.sum())


def test_barycentric_stitcher_no_lower_returns_upper_unchanged() -> None:
    upper = numpy.zeros((8, 8), dtype=float)
    stitcher = BarycentricArrayStitcher(upper=upper)
    stitcher.add_patch(center_x=3.5, center_y=3.5, value=numpy.ones((3, 3)))

    stitched = stitcher.stitch()

    assert stitched is upper
    assert stitched.sum() == pytest.approx(9.0)


def test_barycentric_stitcher_normalizes_by_weights() -> None:
    """With weight tracking, overlapping patches average rather than sum."""
    upper = numpy.zeros((8, 8), dtype=float)
    lower = numpy.zeros((8, 8), dtype=float)
    stitcher = BarycentricArrayStitcher(upper=upper, lower=lower)

    value = 2.0 * numpy.ones((3, 3), dtype=float)
    weight = numpy.ones((3, 3), dtype=float)
    stitcher.add_patch(center_x=3.5, center_y=3.5, value=value, weight=weight)
    stitcher.add_patch(center_x=3.5, center_y=3.5, value=value, weight=weight)

    stitched = stitcher.stitch()

    # Everywhere the weight is > 0, the normalized value must equal 2.0
    covered = lower > 0
    numpy.testing.assert_allclose(stitched[covered], 2.0, atol=1e-12)
    numpy.testing.assert_array_equal(stitched[~covered], 0.0)
