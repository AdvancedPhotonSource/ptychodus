from scipy import ndimage
from scipy.fft import fft, fft2, fftfreq, ifft, ifft2
from scipy.signal.windows import gaussian as gaussian_window
from typing import Literal
import numpy

from .common import ComplexArrayType, InexactArrayType, RealArrayType


class PhaseUnwrapper:
    def __init__(
        self,
        fourier_shift_step: float = 0.5,
        image_grad_method: Literal[
            'fourier_shift', 'fourier_differentiation', 'nearest'
        ] = 'fourier_differentiation',
        image_integration_method: Literal['fourier', 'discrete', 'deconvolution'] = 'fourier',
        weight_map: ComplexArrayType | None = None,
        eps: float = 1e-9,
    ) -> None:
        """Get the unwrapped phase of a complex 2D image.

        Parameters
        ----------
        fourier_shift_step : float
            The finite-difference step size used to calculate the gradient,
            if the Fourier shift method is used.
        image_grad_method : str
            The method used to calculate the phase gradient.
                - "fourier_shift": Use Fourier shift to perform shift.
                - "nearest": Use nearest neighbor to perform shift.
                - "fourier_differentiation": Use Fourier differentiation.
        image_integration_method : str
            The method used to integrate the image back from gradients.
                - "fourier": Use Fourier integration as implemented in PtychoShelves.
                - "deconvolution": Deconvolve ramp filter.
                - "discrete": Use cumulative sum.
        weight_map : ComplexArrayType | None
            A weight map multiplied to the input image.
        eps : float
            A small number to avoid division by zero.
        """
        self.fourier_shift_step = fourier_shift_step
        self.image_grad_method: Literal['fourier_shift', 'fourier_differentiation', 'nearest'] = (
            image_grad_method
        )
        self.image_integration_method: Literal['fourier', 'discrete', 'deconvolution'] = (
            image_integration_method
        )
        self.weight_map = weight_map
        self.eps = eps

    def unwrap(self, img: ComplexArrayType) -> RealArrayType:
        """Run unwrapping.

        Parameters
        ----------
        img : ComplexArrayType
            A 2D complex array giving the image to be unwrapped.

        Returns
        -------
        RealArrayType
            A 2D real array giving the unwrapped phase of the input image.
        """
        if not numpy.iscomplexobj(img):
            raise ValueError('Input array must be complex.')

        if self.weight_map is not None:
            weight_map = float(numpy.clip(self.weight_map, 0.0, 1.0))
        else:
            weight_map = 1.0

        img = weight_map * img / (numpy.abs(img) + self.eps)
        bc_center = float(numpy.angle(img[img.shape[0] // 2, img.shape[1] // 2]))

        # Pad image to avoid FFT boundary artifacts.
        padding = [64, 64]
        if any(numpy.array(padding) > 0):
            img = numpy.pad(
                img, ((padding[0], padding[0]), (padding[1], padding[1])), mode='reflect'
            )
            img = vignette(img, margin=10, sigma=2.5)

        gy, gx = get_phase_gradient(
            img,
            fourier_shift_step=self.fourier_shift_step,
            image_grad_method=self.image_grad_method,
        )

        if self.image_integration_method == 'discrete' and any(numpy.array(padding) > 0):
            gy = gy[padding[0] : -padding[0], padding[1] : -padding[1]]
            gx = gx[padding[0] : -padding[0], padding[1] : -padding[1]]
        if self.image_integration_method == 'discrete':
            phase = numpy.real(integrate_image_2d(gy, gx, bc_center=bc_center))
        elif self.image_integration_method == 'fourier':
            phase = numpy.real(integrate_image_2d_fourier(gy, gx))
        elif self.image_integration_method == 'deconvolution':
            phase = numpy.real(integrate_image_2d_deconvolution(gy, gx, bc_center=bc_center))
        else:
            raise ValueError(f'Unknown integration method: {self.image_integration_method}')

        if self.image_integration_method != 'discrete' and any(numpy.array(padding) > 0):
            gy = gy[padding[0] : -padding[0], padding[1] : -padding[1]]
            gx = gx[padding[0] : -padding[0], padding[1] : -padding[1]]
            phase = phase[padding[0] : -padding[0], padding[1] : -padding[1]]

        return phase


def vignette(img: ComplexArrayType, margin: int = 20, sigma: float = 1.0) -> ComplexArrayType:
    """Vignette an image so that it gradually decays near the boundary.
    For each dimension of the image, a mask with a width of `2 * margin`
    and with half of it filled with 0s and half with 1s is
    generated and convolved with a Gaussian kernel of size
    `margin // 2` and standard deviation `sigma`. The blurred mask is cropped
    and multiplied to the near-edge regions of the image.

    Parameters
    ----------
    img : ComplexArrayType
        The input image.
    margin : int
        The margin of image where the decay takes place.
    sigma : float
        The standard deviation of the Gaussian kernel.

    Returns
    -------
    ComplexArrayType
        The vignetted image.
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


def nearest_neighbor_gradient(
    image: ComplexArrayType,
    direction: Literal['forward', 'backward'],
    dim: int | tuple[int, ...] = (0, 1),
) -> tuple[ComplexArrayType | None, ComplexArrayType | None]:
    """
    Calculate the nearest neighbor gradient of a 2D image.

    Parameters
    ----------
    image : ComplexArrayType
        a (... H, W) tensor of images.
    direction : str
        'forward' or 'backward'.
    dim : int | tuple[int, ...], optional
        Dimensions to calculate gradient. Default is (0, 1).

    Returns
    -------
    tuple[ComplexArrayType | None, ComplexArrayType | None]
        A tuple of 2 images with the gradient in y and x directions.
        Elements are None if the corresponding dimension is not in `dim`.
    """
    dims: tuple[int, ...] = (dim,) if isinstance(dim, int) else dim
    grad_x: ComplexArrayType | None = None
    grad_y: ComplexArrayType | None = None
    if direction == 'forward':
        if 1 in dims:
            grad_x = numpy.concatenate([image[:, 1:], image[:, -1:]], axis=1) - image
        if 0 in dims:
            grad_y = numpy.concatenate([image[1:, :], image[-1:, :]], axis=0) - image
    elif direction == 'backward':
        if 1 in dims:
            grad_x = image - numpy.concatenate([image[:, :1], image[:, :-1]], axis=1)
        if 0 in dims:
            grad_y = image - numpy.concatenate([image[:1, :], image[:-1, :]], axis=0)
    return grad_y, grad_x


def gaussian_gradient(
    image: RealArrayType, sigma: float = 1.0, kernel_size: int = 5
) -> tuple[RealArrayType, RealArrayType]:
    """
    Calculate the gradient of a 2D image with a Gaussian-derivative kernel.

    Parameters
    ----------
    image : RealArrayType
        A (... H, W) tensor of images.
    sigma : float
        Sigma of the Gaussian derivative kernel.
    kernel_size : int
        Size of the Gaussian derivative kernel.

    Returns
    -------
    tuple of RealArrayType
        A tuple of 2 images with the gradient in y and x directions.
    """
    r = numpy.arange(kernel_size) - (kernel_size - 1) / 2.0
    kernel = -r / (numpy.sqrt(2 * numpy.pi) * sigma**3) * numpy.exp(-(r**2) / (2 * sigma**2))
    grad_y = ndimage.convolve(image, kernel.reshape(-1, 1), mode='nearest')
    grad_x = ndimage.convolve(image, kernel.reshape(1, -1), mode='nearest')

    # Gate the gradients
    grads = [grad_y, grad_x]
    for i, g in enumerate(grads):
        m = numpy.logical_and(numpy.abs(grad_y) < 1e-6, numpy.abs(grad_y) != 0)
        if numpy.count_nonzero(m) > 0:
            print('Gradient magnitudes between 0 and 1e-6 are set to 0.')
            g = g * numpy.logical_not(m)
            grads[i] = g
    grad_y, grad_x = grads
    return grad_y, grad_x


def fourier_gradient(image: ComplexArrayType) -> tuple[ComplexArrayType, ComplexArrayType]:
    """Calculate the Fourier-differentiation gradient of an image.

    Multiplies the FFT of the image along each axis by the corresponding
    imaginary ramp filter (2πiu or 2πiv), then inverse-transforms. The
    result is the exact gradient of the band-limited interpolant of the
    input.
    """
    u = fftfreq(image.shape[0])
    v = fftfreq(image.shape[1])
    u, v = numpy.meshgrid(u, v, indexing='ij')

    grad_y = ifft(fft(image, axis=-2) * (2j * numpy.pi * u), axis=-2)
    grad_x = ifft(fft(image, axis=-1) * (2j * numpy.pi * v), axis=-1)

    return grad_y, grad_x


def get_phase_gradient(
    img: ComplexArrayType,
    fourier_shift_step: float = 0,
    image_grad_method: Literal[
        'fourier_shift', 'fourier_differentiation', 'nearest'
    ] = 'fourier_shift',
    _eps: float = 1e-6,
) -> tuple[RealArrayType, RealArrayType]:
    """
    Get the gradient of the phase of a complex 2D image by first calculating
    the spatial gradient of the complex image, then taking the phase of the
    complex gradient -- i.e., it takes the phase of the gradient rather than
    the gradient of the phase. This avoids the sharp gradients due to phase
    wrapping when directly taking the gradient of the phase.

    Parameters
    ----------
    img : ComplexArrayType
        A [H, W] array giving a single image.
    fourier_shift_step : float
        The finite-difference step size used to calculate the gradient, if
        the Fourier shift method is used.
    image_grad_method : str
        The method used to calculate the phase gradient.
            - "fourier_shift": Use Fourier shift to perform shift.
            - "nearest": Use nearest neighbor to perform shift.
            - "fourier_differentiation": Use Fourier differentiation.
    _eps : float
        A stabilizing constant (currently unused).

    Returns
    -------
    tuple[RealArrayType, RealArrayType]
        A tuple of 2 real arrays with the phase gradient in y and x directions.
    """
    if fourier_shift_step <= 0 and image_grad_method == 'fourier_shift':
        raise ValueError('Step must be positive.')

    if image_grad_method == 'fourier_differentiation':
        gy, gx = fourier_gradient(img)
        gy = numpy.imag(numpy.conj(img) * gy)
        gx = numpy.imag(numpy.conj(img) * gx)
    else:
        # Use finite difference.
        if img.ndim == 2:
            img = img[None, ...]
        pad = int(numpy.ceil(fourier_shift_step)) + 1
        img = numpy.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')

        sy1 = numpy.array([[-fourier_shift_step, 0]]).repeat(img.shape[0], axis=0)
        sy2 = numpy.array([[fourier_shift_step, 0]]).repeat(img.shape[0], axis=0)
        if image_grad_method == 'fourier_shift':
            # If the image contains zero-valued pixels, Fourier shift can result in small
            # non-zero values that dangles around 0. This can cause the phase
            # of the shifted image to dangle between pi and -pi. In that case, use
            # `image_grad_method='nearest'` instead, or use `fourier_shift_step=1`.
            complex_prod = fourier_shift(img, sy1) * fourier_shift(img, sy2).conj()
        elif image_grad_method == 'nearest':
            complex_prod = img * numpy.concatenate([img[:, :1, :], img[:, :-1, :]], axis=1).conj()
        complex_prod = numpy.where(
            numpy.abs(complex_prod) < numpy.abs(complex_prod).max() * 1e-6, 0, complex_prod
        )
        gy = numpy.angle(complex_prod) / (2 * fourier_shift_step)
        gy = gy[0, pad:-pad, pad:-pad]

        sx1 = numpy.array([[0, -fourier_shift_step]]).repeat(img.shape[0], axis=0)
        sx2 = numpy.array([[0, fourier_shift_step]]).repeat(img.shape[0], axis=0)
        if image_grad_method == 'fourier_shift':
            complex_prod = fourier_shift(img, sx1) * fourier_shift(img, sx2).conj()
        elif image_grad_method == 'nearest':
            complex_prod = img * numpy.concatenate([img[:, :, :1], img[:, :, :-1]], axis=2).conj()
        complex_prod = numpy.where(
            numpy.abs(complex_prod) < numpy.abs(complex_prod).max() * 1e-6, 0, complex_prod
        )
        gx = numpy.angle(complex_prod) / (2 * fourier_shift_step)
        gx = gx[0, pad:-pad, pad:-pad]

    return gy, gx  # type: ignore


def integrate_image_2d_fourier(
    grad_y: InexactArrayType, grad_x: InexactArrayType
) -> InexactArrayType:
    """
    Integrate an image with the gradient in y and x directions using Fourier
    integration.

    Parameters
    ----------
    grad_y, grad_x: InexactArrayType
        A (H, W) array of gradients in y or x directions.

    Returns
    -------
    InexactArrayType
        The integrated image. Returns the real part if inputs are real.
    """
    shape = grad_y.shape
    f = fft2(grad_x + 1j * grad_y)
    y = fftfreq(shape[0])
    x = fftfreq(shape[1])

    r = 1.0 / (2j * numpy.pi * (x + 1j * y[:, None]) + 1e-15)
    r[0, 0] = 0
    integrated_image = ifft2(f * r)

    return integrated_image if numpy.iscomplexobj(grad_x) else integrated_image.real


def integrate_image_2d_deconvolution(
    grad_y: InexactArrayType,
    grad_x: InexactArrayType,
    tf_y: ComplexArrayType | None = None,
    tf_x: ComplexArrayType | None = None,
    bc_center: float = 0,
) -> ComplexArrayType:
    """
    Integrate an image with the gradient in y and x directions by deconvolving
    the differentiation kernel, whose transfer function is assumed to be a
    ramp function.

    Adapted from Tripathi, A., McNulty, I., Munson, T., & Wild, S. M. (2016).
    Single-view phase retrieval of an extended sample by exploiting edge detection
    and sparsity. Optics Express, 24(21), 24719–24738. doi:10.1364/OE.24.024719

    Parameters
    ----------
    grad_y, grad_x: InexactArrayType
        A (H, W) array of gradients in y or x directions.
    tf_y, tf_x: ComplexArrayType | None
        A (H, W) tensor of transfer functions in y or x directions. If not
        provided, they are assumed to be 2i * pi * u (or v), which are the
        effective transfer functions in Fourier differentiation.
    bc_center: float
        The value of the boundary condition at the center of the image.

    Returns
    -------
    ComplexArrayType
        The integrated image.
    """
    u, v = fftfreq(grad_x.shape[0]), fftfreq(grad_x.shape[1])
    u, v = numpy.meshgrid(u, v, indexing='ij')
    if tf_y is None or tf_x is None:
        tf_y = 2j * numpy.pi * u
        tf_x = 2j * numpy.pi * v
    f_grad_y = fft2(grad_y)
    f_grad_x = fft2(grad_x)
    img = (f_grad_y * tf_y + f_grad_x * tf_x) / (numpy.abs(tf_y) ** 2 + numpy.abs(tf_x) ** 2 + 1e-5)
    img = -ifft2(img)
    img = img + bc_center - img[img.shape[0] // 2, img.shape[1] // 2]
    return img


def integrate_image_2d(
    grad_y: InexactArrayType, grad_x: InexactArrayType, bc_center: float = 0
) -> InexactArrayType:
    """
    Integrate an image with the gradient in y and x directions.

    Parameters
    ----------
    grad_y : InexactArrayType
        The gradient in y direction.
    grad_x : InexactArrayType
        The gradient in x direction.
    bc_center : float
        The value at the center pixel of the integrated image, by default 0.
        Integration uses the left column of grad_y as the y boundary and
        cumulative sums of grad_x along each row.

    Returns
    -------
    InexactArrayType
        The integrated image.
    """
    left_boundary = numpy.cumsum(grad_y[:, 0], axis=0)
    int_img = numpy.cumsum(grad_x, axis=1) + left_boundary[:, None]
    int_img = int_img + bc_center - int_img[int_img.shape[0] // 2, int_img.shape[1] // 2]
    return int_img


def fourier_shift(
    images: ComplexArrayType, shifts: RealArrayType, strictly_preserve_zeros: bool = False
) -> InexactArrayType:
    """
    Apply Fourier shift to a batch of images.

    Parameters
    ----------
    images : ComplexArrayType
        A [N, H, W] array of images.
    shifts : RealArrayType
        A [N, 2] array of shifts in pixels.
    strictly_preserve_zeros : bool
        If True, mask of strictly zero pixels will be generated and shifted
        by the same amount. Pixels that have a non-zero value in the shifted
        mask will be set to zero in the shifted image. This preserves the zero
        pixels in the original image, preventing FFT from introducing small
        non-zero values due to machine precision.

    Returns
    -------
    InexactArrayType
        Shifted images.
    """
    zero_mask_shifted: InexactArrayType | None = None
    if strictly_preserve_zeros:
        zero_mask = (images == 0).astype(float)
        zero_mask_shifted = fourier_shift(zero_mask, shifts, strictly_preserve_zeros=False)
    ft_images = fft2(images)
    freq_y, freq_x = numpy.meshgrid(
        fftfreq(images.shape[-2]), fftfreq(images.shape[-1]), indexing='ij'
    )
    freq_x = freq_x.repeat(images.shape[0], axis=0)
    freq_y = freq_y.repeat(images.shape[0], axis=0)
    mult = numpy.exp(
        1j
        * -2
        * numpy.pi
        * (freq_x * shifts[:, 1].reshape([-1, 1, 1]) + freq_y * shifts[:, 0].reshape([-1, 1, 1]))
    )
    ft_images = ft_images * mult
    shifted_images: InexactArrayType = numpy.asarray(ifft2(ft_images))
    if not numpy.iscomplexobj(images):
        shifted_images = shifted_images.real
    if zero_mask_shifted is not None:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images
