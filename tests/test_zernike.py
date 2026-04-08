import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import ZernikeMonomial


def test_spatial_frequency() -> None:
    assert ZernikeMonomial(1.0, 0, 0).spatial_frequencey == 0
    assert ZernikeMonomial(1.0, 1, 1).spatial_frequencey == 2
    assert ZernikeMonomial(1.0, 1, -1).spatial_frequencey == 2
    assert ZernikeMonomial(1.0, 2, 0).spatial_frequencey == 2
    assert ZernikeMonomial(1.0, 2, 2).spatial_frequencey == 4
    assert ZernikeMonomial(1.0, 3, -1).spatial_frequencey == 4


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
