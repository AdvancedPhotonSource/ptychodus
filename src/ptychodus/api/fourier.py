"""FFT-based operations on 2D arrays: gradient via Fourier differentiation and sub-pixel translation via a phase ramp."""

from scipy.fft import fft, fft2, fftfreq, ifft, ifft2
import numpy

from .constants import TWO_PI_J
from .geometry import PixelGeometry
from .typing import ComplexArrayType, RealArrayType

__all__ = [
    'fourier_gradient',
    'fourier_shift_2d',
]


def fourier_gradient(
    image: ComplexArrayType, pixel_geometry: PixelGeometry | None = None
) -> tuple[ComplexArrayType, ComplexArrayType]:
    """Calculate the Fourier-differentiation gradient of an image.

    If ``pixel_geometry`` is provided, the returned gradient is in units of
    ``image_units / m``; otherwise it is in ``image_units / pixel``.
    """
    dy = pixel_geometry.height_m if pixel_geometry is not None else 1.0
    dx = pixel_geometry.width_m if pixel_geometry is not None else 1.0

    u = fftfreq(image.shape[-2], d=dy).reshape(-1, 1)
    v = fftfreq(image.shape[-1], d=dx)

    grad_y = ifft(fft(image, axis=-2) * (TWO_PI_J * u), axis=-2)
    grad_x = ifft(fft(image, axis=-1) * (TWO_PI_J * v), axis=-1)

    return grad_y, grad_x


def fourier_shift_2d(
    array: ComplexArrayType, dx: float | RealArrayType, dy: float | RealArrayType
) -> ComplexArrayType:
    """Translate the last two axes of ``array`` by ``(dx, dy)`` pixels via
    a Fourier phase ramp. Subpixel shifts are exact for bandlimited signals; positive
    shifts move features toward larger x/y indices. Leading axes are treated as a
    batch dimension. The output preserves the input's dtype.

    Array-valued shifts give each batch member its own: shaped like the leading axes
    with two trailing singleton axes, ``(n, 1, 1)`` for an ``(n, H, W)`` stack, they
    broadcast against the frame."""
    height_px, width_px = array.shape[-2:]
    fy = fftfreq(height_px).reshape(-1, 1)
    fx = fftfreq(width_px)
    # Separable, so two one-dimensional exponentials replace one over every pixel.
    phase = numpy.exp(-TWO_PI_J * fy * dy) * numpy.exp(-TWO_PI_J * fx * dx)
    shifted = ifft2(fft2(array, axes=(-2, -1)) * phase, axes=(-2, -1))
    return shifted.astype(array.dtype)
