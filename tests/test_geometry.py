import numpy
import numpy.testing
import pytest

import scipy.special

from ptychodus.api.geometry import (
    Box2D,
    HermiteMode,
    ImageExtent,
    Interval,
    Line2D,
    PixelGeometry,
    Point2D,
    ZernikeMode,
)
from ptychodus.api.preprocess.probe_positions import AffineTransform


def test_str() -> None:
    assert str(ZernikeMode(1.0, 0, 0)) == '1.0$Z_{0}^{+0}$'
    assert str(ZernikeMode(2.5, 3, -1)) == '2.5$Z_{3}^{-1}$'
    assert str(ZernikeMode(0.5, 2, 2)) == '0.5$Z_{2}^{+2}$'


def test_domain_masking() -> None:
    """Points with distance <= 0 or distance > 1 return undefined_value."""
    mode = ZernikeMode(1.0, 0, 0)
    angle = numpy.zeros(4)
    distance = numpy.array([0.0, -0.5, 1.1, 2.0])
    result = mode(distance, angle, undefined_value=numpy.nan)
    assert numpy.all(numpy.isnan(result))


def test_boundary_included() -> None:
    """distance == 1 is within the domain."""
    mode = ZernikeMode(1.0, 0, 0)
    distance = numpy.array([1.0])
    angle = numpy.array([0.0])
    result = mode(distance, angle, undefined_value=numpy.nan)
    assert numpy.isfinite(result[0])


@pytest.mark.parametrize(
    'radial_degree,angular_frequency,distance,angle,expected',
    [
        # Z_0^0 (piston): sqrt(1) * 1 * 1 = 1
        (0, 0, numpy.array([0.3, 0.7, 1.0]), numpy.array([0.0, 1.0, 2.0]), numpy.ones(3)),
        # Z_1^+1 (tilt): sqrt(4) * r * cos(theta) = 2r*cos(theta)
        (
            1,
            1,
            numpy.array([0.5, 1.0]),
            numpy.array([0.0, numpy.pi / 3]),
            2.0 * numpy.array([0.5, 1.0]) * numpy.cos(numpy.array([0.0, numpy.pi / 3])),
        ),
        # Z_1^-1 (tip): sqrt(4) * r * sin(theta) = 2r*sin(theta)
        (
            1,
            -1,
            numpy.array([0.5, 1.0]),
            numpy.array([numpy.pi / 6, numpy.pi / 4]),
            2.0 * numpy.array([0.5, 1.0]) * numpy.sin(numpy.array([numpy.pi / 6, numpy.pi / 4])),
        ),
        # Z_2^0 (defocus): sqrt(3) * (2r^2 - 1)
        (
            2,
            0,
            numpy.array([0.5, 1.0 / numpy.sqrt(2)]),
            numpy.zeros(2),
            numpy.sqrt(3) * (2 * numpy.array([0.5, 1.0 / numpy.sqrt(2)]) ** 2 - 1),
        ),
        # Z_2^+2: sqrt(6) * r^2 * cos(2*theta)
        (
            2,
            2,
            numpy.array([0.5, 0.8]),
            numpy.array([numpy.pi / 4, numpy.pi / 6]),
            numpy.sqrt(6)
            * numpy.array([0.5, 0.8]) ** 2
            * numpy.cos(2 * numpy.array([numpy.pi / 4, numpy.pi / 6])),
        ),
        # Z_2^-2: sqrt(6) * r^2 * sin(2*theta)
        (
            2,
            -2,
            numpy.array([0.5, 0.8]),
            numpy.array([numpy.pi / 4, numpy.pi / 6]),
            numpy.sqrt(6)
            * numpy.array([0.5, 0.8]) ** 2
            * numpy.sin(2 * numpy.array([numpy.pi / 4, numpy.pi / 6])),
        ),
    ],
)
def test_known_values(
    radial_degree: int,
    angular_frequency: int,
    distance: numpy.ndarray,
    angle: numpy.ndarray,
    expected: numpy.ndarray,
) -> None:
    mode = ZernikeMode(1.0, radial_degree, angular_frequency)
    numpy.testing.assert_allclose(mode(distance, angle), expected, atol=1e-12)


def test_coefficient_scaling() -> None:
    """Output scales linearly with coefficient."""
    mode_unit = ZernikeMode(1.0, 2, 0)
    mode_scaled = ZernikeMode(3.7, 2, 0)
    distance = numpy.array([0.3, 0.6, 0.9])
    angle = numpy.zeros(3)
    numpy.testing.assert_allclose(
        mode_scaled(distance, angle),
        3.7 * mode_unit(distance, angle),
    )


@pytest.mark.parametrize(
    'radial_degree,angular_frequency',
    [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3)],
)
def test_normalization(radial_degree: int, angular_frequency: int) -> None:
    """Integral of Z^2 over the unit disk equals pi (OSA/ANSI normalization)."""
    mode = ZernikeMode(1.0, radial_degree, angular_frequency)
    num_pixels = 512
    Y, X = numpy.mgrid[:num_pixels, :num_pixels]  # noqa: N806
    X = (X - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806
    Y = (Y - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806
    distance = numpy.hypot(Y, X)
    angle = numpy.arctan2(Y, X)
    Z = mode(distance, angle, undefined_value=0.0)  # noqa: N806
    pixel_area = (2.0 / num_pixels) ** 2
    integral = numpy.sum(Z**2) * pixel_area
    numpy.testing.assert_allclose(integral, numpy.pi, rtol=0.01)


def test_indexing() -> None:
    idx = 0

    for n in range(10):
        print('')

        for m in range(-n, n + 1, 2):
            idx_calc = (n * (n + 2) + m) // 2
            print(f'{n=} {m=:+d} {idx=} {idx_calc=}')
            assert idx == idx_calc
            idx += 1


def test_pyramid() -> None:
    import numpy
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.colors
    import matplotlib.pyplot as plt

    from ptychodus.api.geometry import ZernikeMode

    my_dpi = 300
    num_pixels = 256
    max_radial_degree = 6

    Y, X = numpy.mgrid[:num_pixels, :num_pixels]  # noqa: N806
    X = (X - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806
    Y = (Y - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806

    distance = numpy.hypot(Y, X)
    angle = numpy.arctan2(Y, X)

    ###

    fig = plt.figure(dpi=my_dpi)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(max_radial_degree + 1, 2 * (max_radial_degree + 1))

    for radial_degree in range(max_radial_degree):
        for angular_frequency in range(-radial_degree, radial_degree + 1, 2):
            mode = ZernikeMode(1.0, radial_degree, angular_frequency)
            Z = mode(distance, angle, undefined_value=numpy.nan)  # noqa: N806

            row = radial_degree
            col = max_radial_degree + angular_frequency

            ax = fig.add_subplot(gs[row : row + 1, col : col + 2])
            ax.pcolormesh(X, Y, Z, norm=matplotlib.colors.CenteredNorm(), cmap='seismic')
            ax.set_aspect('equal')
            ax.set_title(str(mode))
            ax.axis('off')

    plt.savefig('zernike_pyramid.png', bbox_inches='tight', dpi=my_dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HermiteMode
# ---------------------------------------------------------------------------


def test_hermite_str() -> None:
    assert str(HermiteMode(1.0, 0, 0)) == '1.0$H_{0,0}(x,y)$'
    assert str(HermiteMode(2.5, 3, 1)) == '2.5$H_{3,1}(x,y)$'
    assert str(HermiteMode(0.5, 2, 4)) == '0.5$H_{2,4}(x,y)$'


@pytest.mark.parametrize(
    'order_x,order_y,x,y,expected',
    [
        # H_0(x) * H_0(y) = 1 * 1 = 1
        (0, 0, numpy.array([-1.0, 0.0, 2.5]), numpy.array([0.3, -1.7, 4.0]), numpy.ones(3)),
        # H_1(x) * H_0(y) = 2x * 1 = 2x
        (
            1,
            0,
            numpy.array([0.5, -1.0, 3.0]),
            numpy.array([0.7, 0.0, -2.0]),
            2.0 * numpy.array([0.5, -1.0, 3.0]),
        ),
        # H_0(x) * H_1(y) = 1 * 2y = 2y
        (
            0,
            1,
            numpy.array([0.7, 0.0, -2.0]),
            numpy.array([0.5, -1.0, 3.0]),
            2.0 * numpy.array([0.5, -1.0, 3.0]),
        ),
        # H_2(x) * H_0(y) = (4x^2 - 2) * 1
        (
            2,
            0,
            numpy.array([0.5, 1.0, -1.5]),
            numpy.zeros(3),
            4.0 * numpy.array([0.5, 1.0, -1.5]) ** 2 - 2.0,
        ),
        # H_1(x) * H_1(y) = 2x * 2y = 4xy
        (
            1,
            1,
            numpy.array([0.5, -1.0, 2.0]),
            numpy.array([1.5, 0.5, -0.25]),
            4.0 * numpy.array([0.5, -1.0, 2.0]) * numpy.array([1.5, 0.5, -0.25]),
        ),
        # H_3(x) * H_2(y) = (8x^3 - 12x) * (4y^2 - 2)
        (
            3,
            2,
            numpy.array([0.4, -0.8, 1.2]),
            numpy.array([0.6, 1.1, -0.3]),
            (8.0 * numpy.array([0.4, -0.8, 1.2]) ** 3 - 12.0 * numpy.array([0.4, -0.8, 1.2]))
            * (4.0 * numpy.array([0.6, 1.1, -0.3]) ** 2 - 2.0),
        ),
    ],
)
def test_hermite_known_values(
    order_x: int,
    order_y: int,
    x: numpy.ndarray,
    y: numpy.ndarray,
    expected: numpy.ndarray,
) -> None:
    mode = HermiteMode(1.0, order_x, order_y)
    numpy.testing.assert_allclose(mode(x, y), expected, atol=1e-12)


def test_hermite_coefficient_scaling() -> None:
    """Output scales linearly with the (complex) coefficient."""
    mode_unit = HermiteMode(1.0, 2, 1)
    mode_scaled = HermiteMode(3.7 - 1.2j, 2, 1)
    x = numpy.array([-1.0, 0.0, 0.5, 1.5])
    y = numpy.array([0.2, -0.8, 1.1, -0.3])
    numpy.testing.assert_allclose(
        mode_scaled(x, y),
        (3.7 - 1.2j) * mode_unit(x, y),
    )


@pytest.mark.parametrize('order_x,order_y', [(0, 0), (1, 0), (0, 1), (2, 3), (4, 2), (3, 3)])
def test_hermite_separability(order_x: int, order_y: int) -> None:
    """HermiteMode is the tensor product H_m(x) * H_n(y)."""
    mode = HermiteMode(1.0, order_x, order_y)
    x_1d = numpy.linspace(-2.0, 2.0, 11)
    y_1d = numpy.linspace(-1.5, 2.5, 9)
    Y, X = numpy.meshgrid(y_1d, x_1d, indexing='ij')  # noqa: N806
    expected = scipy.special.eval_hermite(order_x, X) * scipy.special.eval_hermite(order_y, Y)
    numpy.testing.assert_allclose(mode(X, Y), expected, atol=1e-12)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_hermite_three_term_recurrence(n: int) -> None:
    """H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x); verified through HermiteMode along x."""
    x = numpy.linspace(-2.0, 2.0, 25)
    y = numpy.zeros_like(x)  # H_0(y) = 1, so the y factor drops out

    mode_n_plus_1 = HermiteMode(1.0, n + 1, 0)
    mode_n = HermiteMode(1.0, n, 0)

    expected_lhs = mode_n_plus_1(x, y)
    if n == 0:
        expected_rhs = 2.0 * x * mode_n(x, y)
    else:
        mode_n_minus_1 = HermiteMode(1.0, n - 1, 0)
        expected_rhs = 2.0 * x * mode_n(x, y) - 2.0 * n * mode_n_minus_1(x, y)

    numpy.testing.assert_allclose(expected_lhs, expected_rhs, atol=1e-10)


def test_hermite_grid() -> None:
    import numpy
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.colors
    import matplotlib.pyplot as plt

    from ptychodus.api.geometry import HermiteMode

    my_dpi = 300
    num_pixels = 256
    max_order = 5

    Y, X = numpy.mgrid[:num_pixels, :num_pixels]  # noqa: N806
    X = (X - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806
    Y = (Y - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806

    fig = plt.figure(dpi=my_dpi)
    fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(max_order + 1, max_order + 1)

    for m in range(max_order + 1):
        for n in range(max_order + 1):
            mode = HermiteMode(1.0, m, n)
            Z = mode(X, Y).real  # noqa: N806

            ax = fig.add_subplot(gs[m, n])
            ax.pcolormesh(X, Y, Z, norm=matplotlib.colors.CenteredNorm(), cmap='seismic')
            ax.set_aspect('equal')
            ax.set_title(str(mode))
            ax.axis('off')

    plt.savefig('hermite_grid.png', bbox_inches='tight', dpi=my_dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------


def test_interval_contains_is_closed_at_both_endpoints() -> None:
    """`item in interval` includes both lower and upper bounds (closed interval)."""
    interval = Interval[float](1.0, 3.0)
    assert 1.0 in interval
    assert 3.0 in interval
    assert 2.0 in interval


def test_interval_contains_excludes_outside_values() -> None:
    interval = Interval[float](1.0, 3.0)
    assert 0.999 not in interval
    assert 3.001 not in interval


def test_interval_contains_single_point_interval() -> None:
    """A degenerate [a, a] interval contains exactly a."""
    interval = Interval[int](5, 5)
    assert 5 in interval
    assert 4 not in interval
    assert 6 not in interval


def test_interval_contains_improper_interval() -> None:
    """An interval where upper < lower contains nothing."""
    interval = Interval[int](5, 3)
    assert 3 not in interval
    assert 4 not in interval
    assert 5 not in interval


def test_interval_from_bounds_orders_arguments() -> None:
    """from_bounds(a, b) returns Interval(min(a,b), max(a,b))."""
    proper = Interval[float].from_bounds(1.0, 5.0)
    assert (proper.lower, proper.upper) == (1.0, 5.0)

    swapped = Interval[float].from_bounds(5.0, 1.0)
    assert (swapped.lower, swapped.upper) == (1.0, 5.0)


def test_interval_from_bounds_at_equal_arguments() -> None:
    """from_bounds(a, a) returns a degenerate single-point interval."""
    interval = Interval[int].from_bounds(7, 7)
    assert interval.lower == 7
    assert interval.upper == 7


def test_interval_clamp() -> None:
    interval = Interval[int](2, 7)
    assert interval.clamp(0) == 2  # below lower
    assert interval.clamp(2) == 2  # at lower
    assert interval.clamp(5) == 5  # interior
    assert interval.clamp(7) == 7  # at upper
    assert interval.clamp(10) == 7  # above upper


def test_interval_repr() -> None:
    assert repr(Interval[int](2, 7)) == 'Interval(2, 7)'
    assert repr(Interval[float](1.5, 3.5)) == 'Interval(1.5, 3.5)'


# ---------------------------------------------------------------------------
# AffineTransform
# ---------------------------------------------------------------------------


def test_affine_transform_identity() -> None:
    identity = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    points = numpy.array([[2.0, 3.0], [-1.5, 4.7]])
    numpy.testing.assert_allclose(identity(points), points)


def test_affine_transform_pure_translation() -> None:
    """a02, a12 contribute the translation."""
    translate = AffineTransform(1.0, 0.0, 4.0, 0.0, 1.0, -2.0)
    points = numpy.array([[0.0, 0.0], [1.0, 1.0]])
    expected = numpy.array([[4.0, -2.0], [5.0, -1.0]])
    numpy.testing.assert_allclose(translate(points), expected)


def test_affine_transform_pure_scaling() -> None:
    """Diagonal entries scale x and y independently."""
    scale = AffineTransform(2.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    points = numpy.array([[1.0, 1.0], [-2.0, 5.0]])
    expected = numpy.array([[2.0, 3.0], [-4.0, 15.0]])
    numpy.testing.assert_allclose(scale(points), expected)


def test_affine_transform_pure_rotation() -> None:
    """A 90 deg CCW rotation maps (1, 0) -> (0, 1) and (0, 1) -> (-1, 0)."""
    cos90 = numpy.cos(numpy.pi / 2)
    sin90 = numpy.sin(numpy.pi / 2)
    rotate = AffineTransform(cos90, -sin90, 0.0, sin90, cos90, 0.0)
    points = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    expected = numpy.array([[0.0, 1.0], [-1.0, 0.0]])
    numpy.testing.assert_allclose(rotate(points), expected, atol=1e-12)


def test_affine_transform_combined_scale_then_translate() -> None:
    """The convention is x' = a00*x + a01*y + a02; verify with a non-trivial mix."""
    transform = AffineTransform(2.0, 0.5, 1.0, -0.5, 3.0, -2.0)
    # x' = 2*x + 0.5*y + 1
    # y' = -0.5*x + 3*y - 2
    points = numpy.array([[4.0, 6.0]])
    expected = numpy.array([[2 * 4.0 + 0.5 * 6.0 + 1.0, -0.5 * 4.0 + 3.0 * 6.0 - 2.0]])
    numpy.testing.assert_allclose(transform(points), expected)


# ---------------------------------------------------------------------------
# PixelGeometry
# ---------------------------------------------------------------------------


def test_pixel_geometry_is_square_true() -> None:
    assert PixelGeometry(width_m=1e-6, height_m=1e-6).is_square


def test_pixel_geometry_is_square_false_for_anisotropic() -> None:
    assert not PixelGeometry(width_m=1e-6, height_m=2e-6).is_square


def test_pixel_geometry_get_area_m2() -> None:
    pg = PixelGeometry(width_m=2e-6, height_m=3e-6)
    assert pg.get_area_m2() == pytest.approx(6e-12)


def test_pixel_geometry_get_aspect_ratio() -> None:
    """Aspect ratio is width / height (matches the dataclass field convention)."""
    assert PixelGeometry(width_m=4.0, height_m=2.0).get_aspect_ratio() == pytest.approx(2.0)
    assert PixelGeometry(width_m=2.0, height_m=4.0).get_aspect_ratio() == pytest.approx(0.5)


def test_pixel_geometry_copy_equals_original() -> None:
    original = PixelGeometry(width_m=1.5e-6, height_m=2.5e-6)
    duplicate = original.copy()
    assert duplicate == original
    assert duplicate is not original


def test_pixel_geometry_equality() -> None:
    """Frozen dataclass equality compares both fields."""
    assert PixelGeometry(1e-6, 1e-6) == PixelGeometry(1e-6, 1e-6)
    assert PixelGeometry(1e-6, 1e-6) != PixelGeometry(1e-6, 2e-6)


def test_pixel_geometry_is_valid_true_for_strictly_positive() -> None:
    assert PixelGeometry(width_m=1e-6, height_m=1e-6).is_valid


def test_pixel_geometry_is_valid_false_when_either_dimension_is_zero() -> None:
    """Zero on either axis is the 'not ready' sentinel returned by ProductGeometry
    before a dataset binds — must be rejected."""
    assert not PixelGeometry(width_m=0.0, height_m=1e-6).is_valid
    assert not PixelGeometry(width_m=1e-6, height_m=0.0).is_valid
    assert not PixelGeometry(width_m=0.0, height_m=0.0).is_valid


def test_pixel_geometry_is_valid_false_for_negative_dimensions() -> None:
    """Negative pixel sizes are physically meaningless — the predicate treats
    them the same as zero."""
    assert not PixelGeometry(width_m=-1e-6, height_m=1e-6).is_valid
    assert not PixelGeometry(width_m=1e-6, height_m=-1e-6).is_valid


# ---------------------------------------------------------------------------
# ImageExtent
# ---------------------------------------------------------------------------


def test_image_extent_get_shape_is_height_first() -> None:
    """get_shape() returns (height, width) -- numpy convention. Regression here
    would silently transpose the world.
    """
    extent = ImageExtent(width_px=128, height_px=64)
    assert extent.get_shape() == (64, 128)


def test_image_extent_equality() -> None:
    assert ImageExtent(64, 32) == ImageExtent(64, 32)
    assert ImageExtent(64, 32) != ImageExtent(32, 64)


def test_image_extent_equality_against_non_image_extent() -> None:
    """The custom __eq__ explicitly returns False for non-ImageExtent objects."""
    extent = ImageExtent(64, 32)
    assert extent != (64, 32)
    assert extent != 'not an extent'
    assert extent != None  # noqa: E711


# ---------------------------------------------------------------------------
# Line2D
# ---------------------------------------------------------------------------


def test_line2d_lerp_at_endpoints() -> None:
    line = Line2D(begin=Point2D(1.0, 2.0), end=Point2D(5.0, 8.0))
    assert line.lerp(0.0) == Point2D(1.0, 2.0)
    assert line.lerp(1.0) == Point2D(5.0, 8.0)


def test_line2d_lerp_at_midpoint() -> None:
    line = Line2D(begin=Point2D(0.0, 0.0), end=Point2D(4.0, 6.0))
    assert line.lerp(0.5) == Point2D(2.0, 3.0)


def test_line2d_lerp_extrapolates_outside_unit_interval() -> None:
    """alpha outside [0, 1] extrapolates beyond the segment endpoints."""
    line = Line2D(begin=Point2D(0.0, 0.0), end=Point2D(2.0, 4.0))
    assert line.lerp(2.0) == Point2D(4.0, 8.0)
    assert line.lerp(-1.0) == Point2D(-2.0, -4.0)


# ---------------------------------------------------------------------------
# Box2D
# ---------------------------------------------------------------------------


def test_box2d_x_properties() -> None:
    box = Box2D(x=10.0, y=20.0, width=4.0, height=6.0)
    assert box.x_begin == 10.0
    assert box.x_center == 12.0
    assert box.x_end == 14.0


def test_box2d_y_properties() -> None:
    box = Box2D(x=10.0, y=20.0, width=4.0, height=6.0)
    assert box.y_begin == 20.0
    assert box.y_center == 23.0
    assert box.y_end == 26.0


def test_box2d_zero_extent_collapses_to_corner() -> None:
    box = Box2D(x=3.0, y=5.0, width=0.0, height=0.0)
    assert box.x_begin == box.x_center == box.x_end == 3.0
    assert box.y_begin == box.y_center == box.y_end == 5.0


def test_box2d_negative_extent_is_arithmetic() -> None:
    """Box2D does no normalization -- negative width pushes x_end below x_begin."""
    box = Box2D(x=10.0, y=10.0, width=-4.0, height=-6.0)
    assert box.x_end == 6.0
    assert box.x_center == 8.0
    assert box.y_end == 4.0
    assert box.y_center == 7.0
