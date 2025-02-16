# mypy: ignore-errors

import numpy
from numpy.typing import NDArray
from scipy import signal, ndimage
from typing import Literal, Optional, Tuple


class PhaseUnwrapper:
    def __init__(
        self,
        fourier_shift_step: float = 0.5,
        image_grad_method: Literal[
            'fourier_shift', 'fourier_differentiation', 'nearest'
        ] = 'fourier_differentiation',
        image_integration_method: Literal['fourier', 'discrete', 'deconvolution'] = 'fourier',
        weight_map: Optional[NDArray] = None,
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
        weight_map : Optional[NDArray]
            A weight map multiplied to the input image.
        eps : float
            A small number to avoid division by zero.
        """
        self.fourier_shift_step = fourier_shift_step
        self.image_grad_method = image_grad_method
        self.image_integration_method = image_integration_method
        self.weight_map = weight_map
        self.eps = eps

    def unwrap(self, img: NDArray) -> NDArray:
        """Run unwrapping.

        Parameters
        ----------
        img : NDArray
            A 2D complex array giving the image to be unwrapped.

        Returns
        -------
        NDArray
            A 2D real array giving the unwrapped phase of the input image.
        """
        if not numpy.iscomplexobj(img):
            raise ValueError('Input array must be complex.')

        if self.weight_map is not None:
            weight_map = float(numpy.clip(self.weight_map, 0.0, 1.0))
        else:
            weight_map = 1.0

        img = weight_map * img / (numpy.abs(img) + self.eps)
        bc_center = numpy.angle(img[img.shape[0] // 2, img.shape[1] // 2])

        # Pad image to avoid FFT boundary artifacts.
        padding = [64, 64]
        if any(numpy.array(padding) > 0):
            img = numpy.pad(
                img, ((padding[0], padding[0]), (padding[1], padding[1])), mode='reflect'
            )
            img = vignett(img, margin=10, sigma=2.5)

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


def vignett(img: NDArray, margin: int = 20, sigma: float = 1.0) -> NDArray:
    """Vignett an image so that it gradually decays near the boundary.
    For each dimension of the image, a mask with a width of `2 * margin`
    and with half of it filled with 0s and half with 1s is
    generated and convolved with a Gaussian kernel of size
    `margin` and standard deviation `sigma`. The blurred mask is cropped and
    multiplied to the near-edge regions of the image.

    Parameters
    ----------
    img : Tensor
        The input image.
    margin : int
        The margin of image where the decay takes place.
    sigma : float
        The standard deviation of the Gaussian kernel.
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

        gauss_win = signal.windows.gaussian(margin // 2, std=sigma)
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
    image: NDArray, direction: Literal['forward', 'backward'], dim: Tuple[int, ...] = (0, 1)
) -> Tuple[NDArray, NDArray]:
    """
    Calculate the nearest neighbor gradient of a 2D image.

    Parameters
    ----------
    image : NDArray
        a (... H, W) tensor of images.
    direction : str
        'forward' or 'backward'.
    dim : tuple of int, optional
        Dimensions to calculate gradient. Default is (0, 1).

    Returns
    -------
    tuple of NDArray
        a tuple of 2 images with the gradient in y and x directions.
    """
    if not hasattr(dim, '__len__'):
        dim = (dim,)
    grad_x = None
    grad_y = None
    if direction == 'forward':
        if 1 in dim:
            grad_x = numpy.concatenate([image[:, 1:], image[:, -1:]], axis=1) - image
        if 0 in dim:
            grad_y = numpy.concatenate([image[1:, :], image[-1:, :]], axis=0) - image
    elif direction == 'backward':
        if 1 in dim:
            grad_x = image - numpy.concatenate([image[:, :1], image[:, :-1]], axis=1)
        if 0 in dim:
            grad_y = image - numpy.concatenate([image[:1, :], image[:-1, :]], axis=0)
    else:
        raise ValueError("direction must be 'forward' or 'backward'")
    return grad_y, grad_x


def gaussian_gradient(image: NDArray, sigma: float = 1.0, kernel_size=5) -> Tuple[NDArray, NDArray]:
    """
    Calculate the gradient of a 2D image with a Gaussian-derivative kernel.

    Parameters
    ----------
    image : NDArray
        A (... H, W) tensor of images.
    sigma : float
        Sigma of the Gaussian.

    Returns
    -------
    tuple of NDArray
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


def fourier_gradient(image: NDArray) -> Tuple[NDArray, NDArray]:
    """Calculate gradient using NumPy FFT operations"""
    u = numpy.fft.fftfreq(image.shape[0])
    v = numpy.fft.fftfreq(image.shape[1])
    u, v = numpy.meshgrid(u, v, indexing='ij')

    grad_y = numpy.fft.ifft(numpy.fft.fft(image, axis=-2) * (2j * numpy.pi * u), axis=-2)
    grad_x = numpy.fft.ifft(numpy.fft.fft(image, axis=-1) * (2j * numpy.pi * v), axis=-1)

    return grad_y, grad_x


def get_phase_gradient(
    img: NDArray,
    fourier_shift_step: float = 0,
    image_grad_method: Literal[
        'fourier_shift', 'fourier_differentiation', 'nearest'
    ] = 'fourier_shift',
    eps: float = 1e-6,
) -> Tuple[NDArray, NDArray]:
    """
    Get the gradient of the phase of a complex 2D image by first calculating
    the spatial gradient of the complex image, then taking the phase of the
    complex gradient -- i.e., it takes the phase of the gradient rather than
    the gradient of the phase. This avoids the sharp gradients due to phase
    wrapping when directly taking the gradient of the phase.

    Parameters
    ----------
    img : NDArray
        A [N, H, W] or [H, W] tensor giving a batch of images or a single image.
    step : float
        The finite-difference step size used to calculate the gradient, if
        the Fourier shift method is used.
    finite_diff_method : enums.ImageGradientMethods
        The method used to calculate the phase gradient.
            - "fourier_shift": Use Fourier shift to perform shift.
            - "nearest": Use nearest neighbor to perform shift.
            - "fourier_differentiation": Use Fourier differentiation.
    eps : float
        A stablizing constant.

    Returns
    -------
    Tuple[NDArray, NDArray]
        A tuple of 2 images with the gradient in y and x directions.
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
            # `finite_diff_method="nearest" instead`, or use `step=1`.
            complex_prod = fourier_shift(img, sy1) * fourier_shift(img, sy2).conj()
        elif image_grad_method == 'nearest':
            complex_prod = img * numpy.concatenate([img[:, :1, :], img[:, :-1, :]], axis=1).conj()
        else:
            raise ValueError(f'Unknown finite-difference method: {image_grad_method}')
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
    return gy, gx


def integrate_image_2d_fourier(grad_y: NDArray, grad_x: NDArray) -> NDArray:
    """
    Integrate an image with the gradient in y and x directions using Fourier
    differentiation.

    Parameters
    ----------
    grad_y, grad_x: NDArray
        A (H, W) tensor of gradients in y or x directions.

    Returns
    -------
    NDArray
        The integrated image.
    """
    shape = grad_y.shape
    f = numpy.fft.fft2(grad_x + 1j * grad_y)
    y, x = numpy.fft.fftfreq(shape[0]), numpy.fft.fftfreq(shape[1])

    r = 1.0
    r = r / (2j * numpy.pi * (x + 1j * y[:, None]) + 1e-15)
    r[0, 0] = 0
    integrated_image = f * r
    integrated_image = numpy.fft.ifft2(integrated_image)
    if not numpy.iscomplexobj(grad_x):
        integrated_image = integrated_image.real
    return integrated_image


def integrate_image_2d_deconvolution(
    grad_y: NDArray,
    grad_x: NDArray,
    tf_y: Optional[NDArray] = None,
    tf_x: Optional[NDArray] = None,
    bc_center: float = 0,
) -> NDArray:
    """
    Integrate an image with the gradient in y and x directions by deconvolving
    the differentiation kernel, whose transfer function is assumed to be a
    ramp function.

    Adapted from Tripathi, A., McNulty, I., Munson, T., & Wild, S. M. (2016).
    Single-view phase retrieval of an extended sample by exploiting edge detection
    and sparsity. Optics Express, 24(21), 24719–24738. doi:10.1364/OE.24.024719

    Parameters
    ----------
    grad_y, grad_x: NDArray
        A (H, W) tensor of gradients in y or x directions.
    tf_y, tf_x: NDArray
        A (H, W) tensor of transfer functions in y or x directions. If not
        provided, they are assumed to be 2i * pi * u (or v), which are the
        effective transfer functions in Fourier differentiation.
    bc_center: float
        The value of the boundary condition at the center of the image.

    Returns
    -------
    NDArray
        The integrated image.
    """
    u, v = numpy.fft.fftfreq(grad_x.shape[0]), numpy.fft.fftfreq(grad_x.shape[1])
    u, v = numpy.meshgrid(u, v, indexing='ij')
    if tf_y is None or tf_x is None:
        tf_y = 2j * numpy.pi * u
        tf_x = 2j * numpy.pi * v
    f_grad_y = numpy.fft.fft2(grad_y)
    f_grad_x = numpy.fft.fft2(grad_x)
    img = (f_grad_y * tf_y + f_grad_x * tf_x) / (numpy.abs(tf_y) ** 2 + numpy.abs(tf_x) ** 2 + 1e-5)
    img = -numpy.fft.ifft2(img)
    img = img + bc_center - img[img.shape[0] // 2, img.shape[1] // 2]
    return img


def integrate_image_2d(grad_y: NDArray, grad_x: NDArray, bc_center: float = 0) -> NDArray:
    """
    Integrate an image with the gradient in y and x directions.

    Parameters
    ----------
    grad_y : NDArray
        The gradient in y direction.
    grad_x : NDArray
        The gradient in x direction.
    bc_center : float
        The boundary condition at the center of the image, by default 0

    Returns
    -------
    NDArray
        The integrated image.
    """
    left_boundary = numpy.cumsum(grad_y[:, 0], axis=0)
    int_img = numpy.cumsum(grad_x, axis=1) + left_boundary[:, None]
    int_img = int_img + bc_center - int_img[int_img.shape[0] // 2, int_img.shape[1] // 2]
    return int_img


def fourier_shift(
    images: NDArray, shifts: NDArray, strictly_preserve_zeros: bool = False
) -> NDArray:
    """
    Apply Fourier shift to a batch of images.

    Parameters
    ----------
    images : NDArray
        A [N, H, W] array of images.
    shifts : NDArray
        A [N, 2] array of shifts in pixels.
    strictly_preserve_zeros : bool
        If True, mask of strictly zero pixels will be generated and shifted
        by the same amount. Pixels that have a non-zero value in the shifted
        mask will be set to zero in the shifted image. This preserves the zero
        pixels in the original image, preventing FFT from introducing small
        non-zero values due to machine precision.

    Returns
    -------
    NDArray
        Shifted images.
    """
    if strictly_preserve_zeros:
        zero_mask = images == 0
        zero_mask = zero_mask.float()
        zero_mask_shifted = fourier_shift(zero_mask, shifts, strictly_preserve_zeros=False)
    ft_images = numpy.fft.fft2(images)
    freq_y, freq_x = numpy.meshgrid(
        numpy.fft.fftfreq(images.shape[-2]), numpy.fft.fftfreq(images.shape[-1]), indexing='ij'
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
    shifted_images = numpy.fft.ifft2(ft_images)
    if not numpy.iscomplexobj(images):
        shifted_images = shifted_images.real
    if strictly_preserve_zeros:
        shifted_images[zero_mask_shifted > 0] = 0
    return shifted_images
