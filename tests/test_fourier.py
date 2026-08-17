import numpy
import numpy.testing
import pytest

from ptychodus.api.fourier import fourier_gradient, fourier_shift_2d
from ptychodus.api.geometry import PixelGeometry


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


# ---------------------------------------------------------------------------
# fourier_shift_2d
# ---------------------------------------------------------------------------


def _gaussian_2d(height: int, width: int, sigma: float = 4.0) -> numpy.ndarray:
    y = numpy.arange(height).reshape(-1, 1) - (height - 1) / 2
    x = numpy.arange(width).reshape(1, -1) - (width - 1) / 2
    return numpy.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(complex)


class TestFourierShift2D:
    def test_zero_shift_is_identity(self) -> None:
        image = _gaussian_2d(32, 32)
        shifted = fourier_shift_2d(image, dx=0.0, dy=0.0)
        numpy.testing.assert_allclose(shifted, image, atol=1e-12)

    def test_integer_shift_matches_numpy_roll(self) -> None:
        """For a bandlimited signal, an integer-pixel Fourier shift equals numpy.roll."""
        image = _gaussian_2d(32, 32)
        shifted = fourier_shift_2d(image, dx=-3.0, dy=2.0)
        expected = numpy.roll(image, shift=(2, -3), axis=(-2, -1))
        numpy.testing.assert_allclose(shifted, expected, atol=1e-10)

    def test_subpixel_shift_preserves_power(self) -> None:
        """Parseval: a unitary FFT-based shift preserves total power."""
        image = _gaussian_2d(32, 32)
        shifted = fourier_shift_2d(image, dx=-0.25, dy=0.5)
        power_in = float(numpy.sum(numpy.abs(image) ** 2))
        power_out = float(numpy.sum(numpy.abs(shifted) ** 2))
        assert power_out == pytest.approx(power_in, rel=1e-10)

    def test_round_trip_returns_original(self) -> None:
        image = _gaussian_2d(32, 32)
        shifted = fourier_shift_2d(image, dx=-1.3, dy=0.7)
        restored = fourier_shift_2d(shifted, dx=1.3, dy=-0.7)
        numpy.testing.assert_allclose(restored, image, atol=1e-10)

    def test_batched_3d_array_applies_uniformly(self) -> None:
        """A (B, H, W) input is shifted independently per batch plane with the same shift."""
        rng = numpy.random.default_rng(0)
        batch = rng.standard_normal((4, 16, 16)) + 1j * rng.standard_normal((4, 16, 16))
        shifted = fourier_shift_2d(batch, dx=-2.0, dy=1.0)
        for i in range(batch.shape[0]):
            numpy.testing.assert_allclose(
                shifted[i],
                fourier_shift_2d(batch[i], dx=-2.0, dy=1.0),
                atol=1e-12,
            )

    def test_preserves_complex64_dtype(self) -> None:
        image = _gaussian_2d(16, 16).astype(numpy.complex64)
        shifted = fourier_shift_2d(image, dx=0.7, dy=0.3)
        assert shifted.dtype == numpy.complex64

    def test_positive_y_shift_moves_feature_to_larger_row_index(self) -> None:
        """Sign convention: a +y shift moves a bandlimited feature toward larger row indices."""
        image = _gaussian_2d(32, 32, sigma=2.0)
        shifted = fourier_shift_2d(image, dx=0.0, dy=3.0)
        peak_row_in = int(numpy.argmax(numpy.abs(image).sum(axis=-1)))
        peak_row_out = int(numpy.argmax(numpy.abs(shifted).sum(axis=-1)))
        assert peak_row_out - peak_row_in == 3
