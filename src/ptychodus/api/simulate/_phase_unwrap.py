"""Phase unwrapping via gradient-domain integration for complex 2D wavefields."""

from scipy import ndimage
from scipy.fft import fft2, fftfreq, ifft2
from scipy.signal.windows import gaussian as gaussian_window
import numpy

from ..constants import TWO_PI_J
from ..fourier import fourier_gradient
from ..typing import ComplexArrayType, InexactArrayType, RealArrayType


def _vignette(img: ComplexArrayType, margin: int = 20, sigma: float = 1.0) -> ComplexArrayType:
    """Vignette an image so that it gradually decays near the boundary.

    For each dimension of the image, a mask with a width of ``2 * margin``
    and with half of it filled with 0s and half with 1s is generated and
    convolved with a Gaussian kernel of size ``margin // 2`` and standard
    deviation ``sigma``. The blurred mask is cropped and multiplied to the
    near-edge regions of the image.
    """
    img = img.copy()
    for i_dim in range(img.ndim):
        if img.shape[i_dim] <= 2 * margin:
            continue

        mask_shape = (
            [img.shape[i] for i in range(i_dim)]
            + [2 * margin]
            + [img.shape[i] for i in range(i_dim + 1, img.ndim)]
        )
        mask = numpy.zeros(mask_shape)
        mask_slicer = [slice(None)] * i_dim + [slice(margin, None)]
        mask[tuple(mask_slicer)] = 1.0

        gauss_win = gaussian_window(margin // 2, std=sigma)
        gauss_win = gauss_win / numpy.sum(gauss_win)
        mask = ndimage.convolve1d(mask, gauss_win, axis=i_dim, mode='constant')
        mask_final_slicer = [slice(None)] * i_dim + [slice(len(gauss_win), len(gauss_win) + margin)]

        mask = mask[tuple(mask_final_slicer)]

        mask = numpy.where(mask < 1e-3, 0, mask)

        slicer = tuple([slice(None)] * i_dim + [slice(0, margin)])
        img[slicer] = img[slicer] * mask

        slicer = tuple([slice(None)] * i_dim + [slice(-margin, None)])
        img[slicer] = img[slicer] * numpy.flip(mask, axis=i_dim)
    return img


def _integrate_image_2d_fourier(
    grad_y: InexactArrayType, grad_x: InexactArrayType
) -> InexactArrayType:
    """Integrate a 2D image from its y and x gradients via Fourier integration."""
    shape = grad_y.shape
    f = fft2(grad_x + 1j * grad_y)
    y = fftfreq(shape[0])
    x = fftfreq(shape[1])

    r = 1.0 / (TWO_PI_J * (x + 1j * y[:, None]) + 1e-15)
    r[0, 0] = 0
    integrated_image = ifft2(f * r)

    return integrated_image if numpy.iscomplexobj(grad_x) else integrated_image.real


class PhaseUnwrapper:
    """Gradient-domain phase unwrapper for complex 2D images.

    Takes the phase of the complex gradient rather than the gradient of the
    phase, which avoids the sharp gradients due to phase wrapping.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def unwrap(self, img: ComplexArrayType) -> RealArrayType:
        """Return the unwrapped phase of a 2D complex image."""
        if not numpy.iscomplexobj(img):
            raise ValueError('Input array must be complex.')

        img = img / (numpy.abs(img) + self.eps)

        # Pad image to avoid FFT boundary artifacts.
        padding = 64
        img = numpy.pad(img, ((padding, padding), (padding, padding)), mode='reflect')
        img = _vignette(img, margin=10, sigma=2.5)

        gy_c, gx_c = fourier_gradient(img)
        gy = numpy.imag(numpy.conj(img) * gy_c)
        gx = numpy.imag(numpy.conj(img) * gx_c)

        phase = numpy.real(_integrate_image_2d_fourier(gy, gx))
        phase = phase[padding:-padding, padding:-padding]

        return phase
