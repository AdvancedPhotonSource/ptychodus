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

    from ptychodus.model.product.probe.zernike import ZernikePolynomial

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
            polynomial = ZernikePolynomial(radial_degree, angular_frequency)
            Z = polynomial(distance, angle, undefined_value=numpy.nan)  # noqa: N806

            row = radial_degree
            col = max_radial_degree + angular_frequency

            ax = fig.add_subplot(gs[row : row + 1, col : col + 2])
            ax.pcolormesh(X, Y, Z, norm=matplotlib.colors.CenteredNorm(), cmap='seismic')
            ax.set_aspect('equal')
            ax.set_title(str(polynomial))
            ax.axis('off')

    plt.savefig('zernike_pyramid.png', bbox_inches='tight', dpi=my_dpi)
    plt.close(fig)
