"""FFT-based operations on 2D arrays: gradient via Fourier differentiation and sub-pixel translation via a phase ramp."""

from scipy.fft import fft, fft2, fftfreq, ifft, ifft2
import numpy

from .constants import TWO_PI_J
from .geometry import PixelGeometry
from .typing import ComplexArrayType


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


def fourier_shift_2d(array: ComplexArrayType, dx: float, dy: float) -> ComplexArrayType:
    """Translate the last two axes of ``array`` by ``(dx, dy)`` pixels via
    a Fourier phase ramp. Subpixel shifts are exact for bandlimited signals; positive
    shifts move features toward larger x/y indices. Leading axes are treated as a
    batch dimension. The output preserves the input's dtype."""
    height_px, width_px = array.shape[-2:]
    fy = fftfreq(height_px).reshape(-1, 1)
    fx = fftfreq(width_px)
    phase = numpy.exp(-TWO_PI_J * (fy * dy + fx * dx))
    shifted = ifft2(fft2(array, axes=(-2, -1)) * phase, axes=(-2, -1))
    return shifted.astype(array.dtype)
