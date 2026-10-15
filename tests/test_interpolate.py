import numpy
import pytest

from ptychodus.api.interpolate import (
    BarycentricArrayInterpolator,
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
    x_lower = 6.3 - 1.5  # center_x - (width - 1) / 2
    y_lower = 5.7 - 1.5
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
    # (width - 1) / 2 = 1.0; pick center so that x_lower and y_lower are exact half-integers,
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


def test_get_patch_raises_when_the_window_runs_off_the_low_edge() -> None:
    interp = BarycentricArrayInterpolator(numpy.zeros((20, 20)))

    with pytest.raises(ValueError, match='window centered at'):
        interp.get_patch(center_x=1.0, center_y=10.0, width=6, height=6)


def test_get_patch_raises_when_the_window_runs_off_the_high_edge() -> None:
    interp = BarycentricArrayInterpolator(numpy.zeros((20, 20)))

    with pytest.raises(ValueError, match='window centered at'):
        interp.get_patch(center_x=17.0, center_y=10.0, width=6, height=6)


def test_get_patch_does_not_wrap_a_negative_lower_bound() -> None:
    """A negative support start is a valid Python slice, so an unchecked window
    reads from the far end of the axis instead of failing."""
    array = numpy.arange(20 * 20, dtype=float).reshape(20, 20)
    interp = BarycentricArrayInterpolator(array)

    with pytest.raises(ValueError):
        interp.get_patch(center_x=1.0, center_y=1.0, width=6, height=6)


def test_zero_offset_at_the_high_edge_returns_a_full_size_patch() -> None:
    """A zero-offset window needs only ``n`` samples, so it fits flush against the
    far edge where an ``n + 1`` support would not."""
    array = numpy.arange(20 * 20, dtype=float).reshape(20, 20)
    interp = BarycentricArrayInterpolator(array)

    patch = interp.get_patch(center_x=16.5, center_y=16.5, width=6, height=6)

    assert patch.shape == (6, 6)
    numpy.testing.assert_array_equal(patch, array[14:20, 14:20])


@pytest.mark.parametrize('center', [2.5 - 1.0e-15, 2.5 + 1.0e-15])
def test_near_integer_offset_is_snapped_at_the_low_edge(center: float) -> None:
    """An exactly-integer coordinate reaches the interpolator a few ULP away from
    one. Flush against an edge, keeping that residue would demand a sample outside
    the array."""
    array = numpy.arange(20 * 20, dtype=float).reshape(20, 20)
    interp = BarycentricArrayInterpolator(array)

    patch = interp.get_patch(center_x=center, center_y=center, width=6, height=6)

    numpy.testing.assert_allclose(patch, array[0:6, 0:6], atol=1e-9)


def test_near_integer_offset_is_snapped_at_the_high_edge() -> None:
    array = numpy.arange(20 * 20, dtype=float).reshape(20, 20)
    interp = BarycentricArrayInterpolator(array)

    patch = interp.get_patch(center_x=16.5 + 1.0e-15, center_y=16.5, width=6, height=6)

    numpy.testing.assert_allclose(patch, array[14:20, 14:20], atol=1e-9)


def test_add_patch_is_transpose_of_get_patch_at_a_zero_offset() -> None:
    """The adjoint identity on the branch a fractional center never reaches."""
    rng = numpy.random.default_rng(7)
    a = rng.standard_normal((16, 20))
    b = rng.standard_normal((5, 6))

    # width 6 and height 5 put both offsets at zero for these centers.
    center_x, center_y = 9.5, 8.0

    forward = BarycentricArrayInterpolator(a)
    lhs = float(numpy.sum(forward.get_patch(center_x, center_y, b.shape[-1], b.shape[-2]) * b))

    scatter_target = numpy.zeros_like(a)
    BarycentricArrayInterpolator(scatter_target).add_patch(center_x, center_y, b)
    rhs = float(numpy.sum(a * scatter_target))

    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_add_patch_is_transpose_of_get_patch_for_complex_arrays() -> None:
    """The adjoint identity on the dtype the object canvas actually uses.

    The scatter weights are real, so the operator has real matrix entries and the
    identity holds on the conjugate inner product without a sign flip: both the
    real and the imaginary part carry over.
    """
    rng = numpy.random.default_rng(11)
    a = rng.standard_normal((16, 20)) + 1j * rng.standard_normal((16, 20))
    b = rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))

    center_x, center_y = 9.3, 7.8

    forward = BarycentricArrayInterpolator(a)
    lhs = complex(
        numpy.sum(forward.get_patch(center_x, center_y, b.shape[-1], b.shape[-2]) * b.conj())
    )

    scatter_target = numpy.zeros_like(a)
    BarycentricArrayInterpolator(scatter_target).add_patch(center_x, center_y, b)
    rhs = complex(numpy.sum(a * scatter_target.conj()))

    assert lhs.real == pytest.approx(rhs.real, rel=1e-12, abs=1e-12)
    assert lhs.imag == pytest.approx(rhs.imag, rel=1e-12, abs=1e-12)


def test_get_patch_preserves_a_leading_axis() -> None:
    """Layer and mode axes ride along; the window applies to the last two."""
    rng = numpy.random.default_rng(13)
    stack = rng.standard_normal((3, 20, 20))

    patch = BarycentricArrayInterpolator(stack).get_patch(9.3, 7.8, 6, 5)

    assert patch.shape == (3, 5, 6)
    for layer in range(stack.shape[0]):
        expected = BarycentricArrayInterpolator(stack[layer]).get_patch(9.3, 7.8, 6, 5)
        numpy.testing.assert_allclose(patch[layer], expected, atol=1e-12)
