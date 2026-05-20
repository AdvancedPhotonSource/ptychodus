import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import (
    AffineTransform,
    Box2D,
    ImageExtent,
    Interval,
    Line2D,
    PixelGeometry,
    Point2D,
    ZernikeMonomial,
    fourier_gradient,
)


def test_str() -> None:
    assert str(ZernikeMonomial(1.0, 0, 0)) == '1.0$Z_{0}^{+0}$'
    assert str(ZernikeMonomial(2.5, 3, -1)) == '2.5$Z_{3}^{-1}$'
    assert str(ZernikeMonomial(0.5, 2, 2)) == '0.5$Z_{2}^{+2}$'


def test_domain_masking() -> None:
    """Points with distance <= 0 or distance > 1 return undefined_value."""
    monomial = ZernikeMonomial(1.0, 0, 0)
    angle = numpy.zeros(4)
    distance = numpy.array([0.0, -0.5, 1.1, 2.0])
    result = monomial(distance, angle, undefined_value=numpy.nan)
    assert numpy.all(numpy.isnan(result))


def test_boundary_included() -> None:
    """distance == 1 is within the domain."""
    monomial = ZernikeMonomial(1.0, 0, 0)
    distance = numpy.array([1.0])
    angle = numpy.array([0.0])
    result = monomial(distance, angle, undefined_value=numpy.nan)
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
    monomial = ZernikeMonomial(1.0, radial_degree, angular_frequency)
    numpy.testing.assert_allclose(monomial(distance, angle), expected, atol=1e-12)


def test_coefficient_scaling() -> None:
    """Output scales linearly with coefficient."""
    monomial_unit = ZernikeMonomial(1.0, 2, 0)
    monomial_scaled = ZernikeMonomial(3.7, 2, 0)
    distance = numpy.array([0.3, 0.6, 0.9])
    angle = numpy.zeros(3)
    numpy.testing.assert_allclose(
        monomial_scaled(distance, angle),
        3.7 * monomial_unit(distance, angle),
    )


@pytest.mark.parametrize(
    'radial_degree,angular_frequency',
    [(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), (3, -3), (3, -1), (3, 1), (3, 3)],
)
def test_normalization(radial_degree: int, angular_frequency: int) -> None:
    """Integral of Z^2 over the unit disk equals pi (OSA/ANSI normalization)."""
    monomial = ZernikeMonomial(1.0, radial_degree, angular_frequency)
    num_pixels = 512
    Y, X = numpy.mgrid[:num_pixels, :num_pixels]  # noqa: N806
    X = (X - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806
    Y = (Y - (num_pixels - 1) / 2) / (num_pixels / 2)  # noqa: N806
    distance = numpy.hypot(Y, X)
    angle = numpy.arctan2(Y, X)
    Z = monomial(distance, angle, undefined_value=0.0)  # noqa: N806
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

    from ptychodus.api.geometry import ZernikeMonomial

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
            monomial = ZernikeMonomial(1.0, radial_degree, angular_frequency)
            Z = monomial(distance, angle, undefined_value=numpy.nan)  # noqa: N806

            row = radial_degree
            col = max_radial_degree + angular_frequency

            ax = fig.add_subplot(gs[row : row + 1, col : col + 2])
            ax.pcolormesh(X, Y, Z, norm=matplotlib.colors.CenteredNorm(), cmap='seismic')
            ax.set_aspect('equal')
            ax.set_title(str(monomial))
            ax.axis('off')

    plt.savefig('zernike_pyramid.png', bbox_inches='tight', dpi=my_dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# fourier_gradient
# ---------------------------------------------------------------------------
#
# fourier_gradient computes ifft(fft(f) * 2*pi*i*nu) along each spatial axis.
# For an input that is itself an FFT basis vector exp(2*pi*i*k*m/N), the
# derivative is analytically 2*pi*i*(k/N)/spacing times the input. These
# tests use that identity to check correctness exactly (up to FP roundoff),
# avoiding any quadrature error.


def _fft_mode_2d(
    height: int, width: int, ky: int, kx: int, dy: float = 1.0, dx: float = 1.0
) -> numpy.ndarray:
    """Sample exp(2*pi*i*(ky*y/H + kx*x/W)) on the H x W grid with given spacing.

    Using physical coordinates y_m = m*dy, x_n = n*dx, the analytical
    derivative wrt y is 2*pi*i*(ky/(H*dy)) times the mode -- so the (ky, kx)
    that selects an FFT basis vector is independent of spacing.
    """
    m = numpy.arange(height).reshape(-1, 1)
    n = numpy.arange(width).reshape(1, -1)
    return numpy.exp(2j * numpy.pi * (ky * m / height + kx * n / width))


def test_fourier_gradient_constant_input_is_zero() -> None:
    """A constant image has zero gradient (only DC bin, multiplied by 0)."""
    image = numpy.full((16, 24), 3.0 + 1.5j, dtype=complex)
    gy, gx = fourier_gradient(image)
    numpy.testing.assert_allclose(gy, 0.0, atol=1e-12)
    numpy.testing.assert_allclose(gx, 0.0, atol=1e-12)


def test_fourier_gradient_shape_and_dtype_match_input() -> None:
    image = numpy.zeros((16, 24), dtype=complex)
    gy, gx = fourier_gradient(image)
    assert gy.shape == image.shape
    assert gx.shape == image.shape
    assert numpy.iscomplexobj(gy)
    assert numpy.iscomplexobj(gx)


@pytest.mark.parametrize(
    'shape,ky,kx',
    [
        ((16, 24), 1, 0),
        ((16, 24), 0, 1),
        ((16, 24), 3, -2),
        ((32, 32), 5, 7),
        ((17, 19), 2, -3),  # odd sizes
    ],
)
def test_fourier_gradient_matches_analytical_for_fourier_modes(
    shape: tuple[int, int], ky: int, kx: int
) -> None:
    """For an FFT basis vector, the gradient is exactly 2*pi*i*nu * f."""
    height, width = shape
    image = _fft_mode_2d(height, width, ky, kx)
    gy, gx = fourier_gradient(image)

    expected_gy = 2j * numpy.pi * (ky / height) * image
    expected_gx = 2j * numpy.pi * (kx / width) * image

    numpy.testing.assert_allclose(gy, expected_gy, atol=1e-10)
    numpy.testing.assert_allclose(gx, expected_gx, atol=1e-10)


def test_fourier_gradient_linearity() -> None:
    """gradient(a*f + b*g) == a*gradient(f) + b*gradient(g)."""
    rng = numpy.random.default_rng(0)
    f = rng.standard_normal((20, 28)) + 1j * rng.standard_normal((20, 28))
    g = rng.standard_normal((20, 28)) + 1j * rng.standard_normal((20, 28))
    a = 2.5 - 1.1j
    b = -0.7 + 0.4j

    gy_f, gx_f = fourier_gradient(f)
    gy_g, gx_g = fourier_gradient(g)
    gy_sum, gx_sum = fourier_gradient(a * f + b * g)

    numpy.testing.assert_allclose(gy_sum, a * gy_f + b * gy_g, atol=1e-12)
    numpy.testing.assert_allclose(gx_sum, a * gx_f + b * gx_g, atol=1e-12)


def test_fourier_gradient_pixel_geometry_isotropic_scaling() -> None:
    """Halving the pixel size doubles the gradient (units = image_units / m)."""
    image = _fft_mode_2d(32, 32, 3, -2)
    gy_unit, gx_unit = fourier_gradient(image)

    pg = PixelGeometry(width_m=0.5, height_m=0.5)
    gy_phys, gx_phys = fourier_gradient(image, pixel_geometry=pg)

    numpy.testing.assert_allclose(gy_phys, gy_unit / 0.5, atol=1e-10)
    numpy.testing.assert_allclose(gx_phys, gx_unit / 0.5, atol=1e-10)


def test_fourier_gradient_pixel_geometry_anisotropic_scaling() -> None:
    """y and x axes scale independently by their respective pixel sizes."""
    image = _fft_mode_2d(32, 24, 4, 3)
    gy_unit, gx_unit = fourier_gradient(image)

    dy, dx = 2.0e-6, 5.0e-7
    pg = PixelGeometry(width_m=dx, height_m=dy)
    gy_phys, gx_phys = fourier_gradient(image, pixel_geometry=pg)

    numpy.testing.assert_allclose(gy_phys, gy_unit / dy, atol=1e-6)
    numpy.testing.assert_allclose(gx_phys, gx_unit / dx, atol=1e-6)


def test_fourier_gradient_pixel_geometry_matches_analytical() -> None:
    """With explicit pixel spacing the result matches the closed-form derivative."""
    height, width = 32, 24
    ky, kx = 4, -3
    dy, dx = 1.5e-6, 2.5e-6
    image = _fft_mode_2d(height, width, ky, kx)

    pg = PixelGeometry(width_m=dx, height_m=dy)
    gy, gx = fourier_gradient(image, pixel_geometry=pg)

    expected_gy = 2j * numpy.pi * (ky / (height * dy)) * image
    expected_gx = 2j * numpy.pi * (kx / (width * dx)) * image

    numpy.testing.assert_allclose(gy, expected_gy, rtol=1e-10, atol=1e-6)
    numpy.testing.assert_allclose(gx, expected_gx, rtol=1e-10, atol=1e-6)


def test_fourier_gradient_preserves_leading_batch_dimensions() -> None:
    """The implementation operates on axes (-2, -1); leading dims pass through."""
    batch = numpy.stack(
        [
            _fft_mode_2d(16, 16, 1, 0),
            _fft_mode_2d(16, 16, 0, 2),
            _fft_mode_2d(16, 16, 3, -1),
        ],
        axis=0,
    )
    assert batch.shape == (3, 16, 16)

    gy, gx = fourier_gradient(batch)
    assert gy.shape == batch.shape
    assert gx.shape == batch.shape

    for i in range(batch.shape[0]):
        gy_i, gx_i = fourier_gradient(batch[i])
        numpy.testing.assert_allclose(gy[i], gy_i, atol=1e-12)
        numpy.testing.assert_allclose(gx[i], gx_i, atol=1e-12)


def test_fourier_gradient_superposition_of_modes() -> None:
    """Sum of modes -> sum of analytical derivatives (linearity + correctness)."""
    height, width = 24, 32
    modes = [(1, 0, 1.0 + 0j), (0, 2, -0.5j), (3, -4, 2.0 - 1.0j)]

    image = numpy.zeros((height, width), dtype=complex)
    expected_gy = numpy.zeros_like(image)
    expected_gx = numpy.zeros_like(image)
    for ky, kx, amp in modes:
        mode = _fft_mode_2d(height, width, ky, kx)
        image = image + amp * mode
        expected_gy = expected_gy + amp * (2j * numpy.pi * (ky / height)) * mode
        expected_gx = expected_gx + amp * (2j * numpy.pi * (kx / width)) * mode

    gy, gx = fourier_gradient(image)
    numpy.testing.assert_allclose(gy, expected_gy, atol=1e-10)
    numpy.testing.assert_allclose(gx, expected_gx, atol=1e-10)


def test_fourier_gradient_default_spacing_matches_unit_pixel_geometry() -> None:
    """Passing pixel_geometry=None should match PixelGeometry(1.0, 1.0)."""
    image = _fft_mode_2d(20, 20, 2, 1)
    gy_default, gx_default = fourier_gradient(image)
    gy_unit, gx_unit = fourier_gradient(
        image, pixel_geometry=PixelGeometry(width_m=1.0, height_m=1.0)
    )
    numpy.testing.assert_allclose(gy_default, gy_unit, atol=1e-12)
    numpy.testing.assert_allclose(gx_default, gx_unit, atol=1e-12)


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


def test_interval_create_proper_orders_arguments() -> None:
    """create_proper(a, b) returns Interval(min(a,b), max(a,b))."""
    proper = Interval[float].create_proper(1.0, 5.0)
    assert (proper.lower, proper.upper) == (1.0, 5.0)

    swapped = Interval[float].create_proper(5.0, 1.0)
    assert (swapped.lower, swapped.upper) == (1.0, 5.0)


def test_interval_create_proper_at_equal_arguments() -> None:
    """create_proper(a, a) returns a degenerate single-point interval."""
    interval = Interval[int].create_proper(7, 7)
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
    assert identity(2.0, 3.0) == (2.0, 3.0)
    assert identity(-1.5, 4.7) == (-1.5, 4.7)


def test_affine_transform_pure_translation() -> None:
    """a02, a12 contribute the translation."""
    translate = AffineTransform(1.0, 0.0, 4.0, 0.0, 1.0, -2.0)
    assert translate(0.0, 0.0) == (4.0, -2.0)
    assert translate(1.0, 1.0) == (5.0, -1.0)


def test_affine_transform_pure_scaling() -> None:
    """Diagonal entries scale x and y independently."""
    scale = AffineTransform(2.0, 0.0, 0.0, 0.0, 3.0, 0.0)
    assert scale(1.0, 1.0) == (2.0, 3.0)
    assert scale(-2.0, 5.0) == (-4.0, 15.0)


def test_affine_transform_pure_rotation() -> None:
    """A 90 deg CCW rotation maps (1, 0) -> (0, 1) and (0, 1) -> (-1, 0)."""
    cos90 = numpy.cos(numpy.pi / 2)
    sin90 = numpy.sin(numpy.pi / 2)
    rotate = AffineTransform(cos90, -sin90, 0.0, sin90, cos90, 0.0)
    xp, yp = rotate(1.0, 0.0)
    assert xp == pytest.approx(0.0, abs=1e-12)
    assert yp == pytest.approx(1.0, abs=1e-12)
    xp, yp = rotate(0.0, 1.0)
    assert xp == pytest.approx(-1.0, abs=1e-12)
    assert yp == pytest.approx(0.0, abs=1e-12)


def test_affine_transform_combined_scale_then_translate() -> None:
    """The convention is x' = a00*x + a01*y + a02; verify with a non-trivial mix."""
    transform = AffineTransform(2.0, 0.5, 1.0, -0.5, 3.0, -2.0)
    # x' = 2*x + 0.5*y + 1
    # y' = -0.5*x + 3*y - 2
    xp, yp = transform(4.0, 6.0)
    assert xp == pytest.approx(2 * 4.0 + 0.5 * 6.0 + 1.0)
    assert yp == pytest.approx(-0.5 * 4.0 + 3.0 * 6.0 - 2.0)


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


# ---------------------------------------------------------------------------
# fourier_gradient — additional coverage
# ---------------------------------------------------------------------------


def test_fourier_gradient_real_input_below_nyquist_yields_real_gradient() -> None:
    """For even N, a real cosine well below Nyquist gives a near-real gradient
    (the conjugate-symmetric ±k bins cancel out the imaginary residue).
    """
    height, width = 32, 32
    k = 5  # safely below Nyquist (16)
    m = numpy.arange(height).reshape(-1, 1).astype(float)
    image = numpy.broadcast_to(numpy.cos(2 * numpy.pi * k * m / height), (height, width))
    gy, _ = fourier_gradient(image)
    assert numpy.max(numpy.abs(gy.imag)) < 1e-10
    expected = -2 * numpy.pi * k / height * numpy.sin(2 * numpy.pi * k * m / height)
    expected = numpy.broadcast_to(expected, (height, width))
    numpy.testing.assert_allclose(gy.real, expected, atol=1e-10)


def test_fourier_gradient_real_input_at_nyquist_is_purely_imaginary() -> None:
    """For even N, a real Nyquist mode (-1)^m produces a *purely imaginary*
    gradient -i*pi*(-1)^m. This documents the Nyquist asymmetry: the
    multiplier 2*pi*i*nu_Nyquist = -i*pi has no conjugate partner, so a real
    input does not produce a real gradient.
    """
    height, width = 16, 16
    m = numpy.arange(height).reshape(-1, 1).astype(float)
    image = numpy.broadcast_to((-1.0) ** m, (height, width)).astype(float)
    gy, _ = fourier_gradient(image)
    expected = -1j * numpy.pi * numpy.broadcast_to((-1.0) ** m, (height, width))
    numpy.testing.assert_allclose(gy, expected, atol=1e-10)


def test_fourier_gradient_higher_rank_batch_dims() -> None:
    """The implementation operates on axes (-2, -1) so any number of leading
    batch dimensions should pass through.
    """
    rng = numpy.random.default_rng(42)
    image = rng.standard_normal((2, 3, 16, 16)) + 1j * rng.standard_normal((2, 3, 16, 16))
    gy, gx = fourier_gradient(image)
    assert gy.shape == image.shape
    assert gx.shape == image.shape
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gy_ij, gx_ij = fourier_gradient(image[i, j])
            numpy.testing.assert_allclose(gy[i, j], gy_ij, atol=1e-12)
            numpy.testing.assert_allclose(gx[i, j], gx_ij, atol=1e-12)


def test_fourier_gradient_odd_shape_anisotropic_spacing() -> None:
    """Odd sizes have no Nyquist bin; verify analytic correctness with non-power
    -of-2 dimensions and anisotropic physical pixel spacing.
    """
    height, width = 17, 31
    ky, kx = 2, 5
    dy, dx = 1.5e-6, 4.0e-6
    image = _fft_mode_2d(height, width, ky, kx)

    pg = PixelGeometry(width_m=dx, height_m=dy)
    gy, gx = fourier_gradient(image, pixel_geometry=pg)

    expected_gy = 2j * numpy.pi * (ky / (height * dy)) * image
    expected_gx = 2j * numpy.pi * (kx / (width * dx)) * image

    numpy.testing.assert_allclose(gy, expected_gy, rtol=1e-10, atol=1e-6)
    numpy.testing.assert_allclose(gx, expected_gx, rtol=1e-10, atol=1e-6)
