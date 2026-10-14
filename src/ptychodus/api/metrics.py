"""Resolution metrics for reconstructed images (Fourier ring correlation, etc.)."""

from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum, auto
import logging

import numpy
import scipy.fft
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.signal.windows import tukey
from skimage.metrics import (
    normalized_mutual_information,
    peak_signal_noise_ratio,
    structural_similarity,
)

from .typing import ComplexArrayType, IntegerArrayType, RealArrayType
from .diffraction import BadPixels, DiffractionPatterns
from .fourier import fourier_shift_2d
from .illumination import IlluminationMap
from .simulate.diffraction import generate_diffraction_data
from .geometry import PixelGeometry
from .object import (
    ObjectCenter,
    RegistrationQuantity,
    center_crop_object,
    estimate_object_alignment_shift,
    shift_object,
)
from .product import Product
from .reconstruct import ReconstructionAmbiguities

logger = logging.getLogger(__name__)


def _validate_weights(
    weights: RealArrayType | None, expected_shape: tuple[int, ...]
) -> RealArrayType | None:
    if weights is None:
        return None
    weights_arr = numpy.asarray(weights, dtype=numpy.float64)
    if weights_arr.shape != expected_shape:
        raise ValueError(
            f'weights shape {weights_arr.shape} does not match'
            f' object layer 0 shape {expected_shape}!'
        )
    if not numpy.all(numpy.isfinite(weights_arr)):
        raise ValueError('weights must all be finite!')
    if numpy.any(weights_arr < 0.0):
        raise ValueError('weights must all be non-negative!')
    return weights_arr


def _estimate_phase_offset_and_ramp(
    *,
    signal: numpy.ndarray,
    weights: RealArrayType | None,
    pixel_width_m: float,
    pixel_height_m: float,
    position_x_m: RealArrayType,
    position_y_m: RealArrayType,
) -> tuple[float, float, float]:
    """Recover (phi, k_x_rad_per_m, k_y_rad_per_m) from a complex 2D signal.

    The signal's phase is assumed to be ``phi + k_x*x + k_y*y`` plus
    high-frequency content; its magnitude provides the natural amplitude
    weighting. The ramp is recovered from per-pixel complex differences along
    each axis (so unwrapping is unnecessary), then ``phi`` is recovered as the
    weighted circular mean of the de-ramped signal.
    """
    # Differential phase along x: arg(S[:, x+1] * conj(S[:, x])) carries
    # k_x * pixel_width_m modulo 2pi without ever wrapping per-pair.
    delta_x = signal[:, 1:] * numpy.conj(signal[:, :-1])
    if weights is None:
        accum_x = numpy.sum(delta_x)
    else:
        w_x = weights[:, 1:] * weights[:, :-1]
        accum_x = numpy.sum(w_x * delta_x)
    k_x_per_px = numpy.angle(accum_x)

    delta_y = signal[1:, :] * numpy.conj(signal[:-1, :])
    if weights is None:
        accum_y = numpy.sum(delta_y)
    else:
        w_y = weights[1:, :] * weights[:-1, :]
        accum_y = numpy.sum(w_y * delta_y)
    k_y_per_px = numpy.angle(accum_y)

    k_x_rad_per_m = k_x_per_px / pixel_width_m
    k_y_rad_per_m = k_y_per_px / pixel_height_m

    ramp = k_x_rad_per_m * position_x_m + k_y_rad_per_m * position_y_m
    signal_deramped = signal * numpy.exp(-1j * ramp)
    if weights is None:
        phi_accum = numpy.sum(signal_deramped)
    else:
        phi_accum = numpy.sum(weights * signal_deramped)

    if phi_accum == 0:
        raise ValueError('Cannot estimate phase offset: weighted signal magnitude is zero.')

    phi = numpy.angle(phi_accum)
    return float(phi), float(k_x_rad_per_m), float(k_y_rad_per_m)


def estimate_reconstruction_ambiguities(
    product: Product,
    *,
    reference: Product | None = None,
    weights: RealArrayType | None = None,
) -> ReconstructionAmbiguities:
    """Estimate the ambiguities present in ``product``.

    Without ``reference``: estimate ``(phi, k_x, k_y)`` that flatten layer
    0's phase in the amplitude-weighted circular-mean sense.
    ``object_scale_factor`` is fixed at ``1.0`` because there is no
    reference amplitude to normalize against.

    With ``reference``: estimate all four ambiguities ``(s, phi, k_x, k_y)``
    on ``product`` such that
    ``estimate.standardize_product(product)`` best matches ``reference`` in
    the weighted least-squares sense. The two products must agree in
    layer-0 shape and object pixel geometry. The driving signal becomes
    ``S = product[0] * conj(reference[0])``, whose phase is exactly
    ``phi + k_x*x + k_y*y`` and whose magnitude ``|product| * |reference|``
    provides natural amplitude weighting (pixels where either product is
    weak contribute little).

    The estimate is fully complex-domain (sums of phasors, ``numpy.angle``
    of complex weighted sums) and so requires no phase unwrapping. Pixels
    of zero amplitude contribute exactly zero to the relevant sums and are
    therefore ignored automatically.

    Args:
        product: Product whose ambiguities are being measured. The result
            is returned in this product's coordinate frame.
        reference: Optional anchor product. When supplied, the scale factor
            is estimated too; when ``None``, scale is fixed at ``1.0``.
        weights: Optional non-negative per-pixel weight array, shape
            ``(height_px, width_px)`` matching layer 0. Multiplies the
            natural amplitude weighting. Pass a 0/1 mask to restrict the
            estimate to a region of interest.
    """
    obj = product.object_
    layer_zero = obj.get_array()[0].astype(numpy.complex128)
    pixel_geometry = obj.get_pixel_geometry()
    coords = obj.get_geometry().get_transverse_coordinates()
    weights_arr = _validate_weights(weights, layer_zero.shape)

    if reference is None:
        ref_layer_zero = None
        signal = layer_zero
    else:
        ref_obj = reference.object_
        ref_shape = ref_obj.get_array().shape[-2:]

        if ref_shape != layer_zero.shape:
            raise ValueError(
                f'Object layer-0 shape mismatch: reference {ref_shape} vs product {layer_zero.shape}!'
            )

        ref_pixel_geometry = ref_obj.get_pixel_geometry()

        if ref_pixel_geometry != pixel_geometry:
            raise ValueError(
                f'Object pixel geometry mismatch: reference {ref_pixel_geometry} vs product {pixel_geometry}!'
            )

        ref_layer_zero = ref_obj.get_array()[0].astype(numpy.complex128)
        signal = layer_zero * numpy.conj(ref_layer_zero)

    phi, k_x, k_y = _estimate_phase_offset_and_ramp(
        signal=signal,
        weights=weights_arr,
        pixel_width_m=pixel_geometry.width_m,
        pixel_height_m=pixel_geometry.height_m,
        position_x_m=coords.x_m,
        position_y_m=coords.y_m,
    )

    if ref_layer_zero is None:
        s = 1.0
    else:
        # Weighted-LS solution for s in product ≈ s * exp(i(phi + k·r)) * ref:
        # s = Re(sum w * signal * exp(-i(phi + ramp))) / sum(w * |ref|^2).
        ramp = k_x * coords.x_m + k_y * coords.y_m
        correction = numpy.exp(-1j * (phi + ramp))
        ref_intensity = numpy.square(numpy.abs(ref_layer_zero))

        if weights_arr is None:
            numerator = numpy.real(numpy.sum(signal * correction))
            denominator = numpy.sum(ref_intensity)
        else:
            numerator = numpy.real(numpy.sum(weights_arr * signal * correction))
            denominator = numpy.sum(weights_arr * ref_intensity)

        if not (denominator > 0.0):
            raise ValueError('Cannot estimate scale: weighted reference object intensity is zero.')

        s = numerator / denominator

        # Convention: keep object_scale_factor > 0. Fold any sign flip into phi.
        if s < 0.0:
            s = -s
            phi = phi + numpy.pi

    return ReconstructionAmbiguities(
        object_scale_factor=float(s),
        phase_offset_rad=phi,
        phase_ramp_x_rad_per_m=k_x,
        phase_ramp_y_rad_per_m=k_y,
    )


@dataclass(frozen=True)
class ObjectComparison:
    """Two ptychography object reconstructions standardized and aligned for metric comparison.

    Reconstructed objects are uniquely determined only up to a global complex scale,
    a 2D linear phase ramp, and a sub-pixel translation (any of which leaves the
    measured diffraction intensities unchanged). A pixelwise quality metric
    (SSIM, PSNR, RMSE, MAE, FRC, ...) computed on raw reconstructions therefore
    reflects ambiguity noise rather than real reconstruction error. This dataclass
    holds the result of removing those degrees of freedom so the same prepared
    pair can be fed to every metric.
    """

    reference_complex: ComplexArrayType
    """2D complex array, the reference object's flattened layers."""
    test_complex: ComplexArrayType
    """2D complex array, standardized + aligned. Same shape and dtype as ``reference_complex``."""
    pixel_geometry: PixelGeometry
    ambiguities: ReconstructionAmbiguities
    """The ambiguities removed from the test side."""

    @property
    def reference_amplitude(self) -> RealArrayType:
        """``|reference|``."""
        return numpy.absolute(self.reference_complex)

    @property
    def test_amplitude(self) -> RealArrayType:
        """``|test|``."""
        return numpy.absolute(self.test_complex)

    @property
    def reference_phase(self) -> RealArrayType:
        """``arg(reference)`` in radians, wrapped to ``(-pi, pi]``."""
        return numpy.angle(self.reference_complex)

    @property
    def test_phase(self) -> RealArrayType:
        """``arg(test)`` in radians, wrapped to ``(-pi, pi]``."""
        return numpy.angle(self.test_complex)


def compute_object_comparison(
    reference: Product,
    test: Product,
    *,
    upsample_factor: int = 100,
    weights: RealArrayType | None = None,
    registration_quantity: RegistrationQuantity = RegistrationQuantity.COMPLEX,
    num_alignment_iterations: int = 4,
    alignment_tolerance_px: float = 0.5,
) -> ObjectComparison:
    """Align ``test`` onto ``reference``, standardize ambiguities, and bundle the pair.

    Translation and the linear phase ramp are coupled: a residual ramp biases
    the registration peak, and a residual translation biases the ramp estimate.
    A single pass through "register, then de-ramp" therefore leaves both wrong.
    This function alternates the two until the registration stops moving. The
    shift is accumulated and applied once to the original test object on every
    pass, so the object is interpolated exactly once no matter how many
    iterations run.

    Args:
        reference: The reconstruction treated as ground truth. Defines the
            array indexing and ambiguity anchor.
        test: The reconstruction being evaluated against ``reference``.
        upsample_factor: Sub-pixel precision for the phase-cross-correlation
            registration.
        weights: Optional non-negative per-pixel weights for the ambiguity
            estimate, shape ``(height_px, width_px)`` matching layer 0. Pass
            a 0/1 mask to restrict the estimate to a region of interest.
        registration_quantity: Which quantity derived from the complex object is
            correlated; see :class:`ptychodus.api.object.RegistrationQuantity`.
            The default avoids ``AMPLITUDE``, which is nearly featureless for a
            phase-contrast sample and so registers on the channel with the
            least signal. Only the refinement passes use this -- the bootstrap
            pass is forced onto a ramp-invariant quantity regardless.
        num_alignment_iterations: Maximum number of register/de-ramp passes.
        alignment_tolerance_px: Stop once a pass moves the test object by less
            than this many pixels.

    Raises:
        ValueError: If the two products' objects disagree on pixel geometry or
            if the weighted reference intensity is zero.
    """
    if num_alignment_iterations < 1:
        raise ValueError(
            f'Number of alignment iterations must be positive; got {num_alignment_iterations}!'
        )

    reference_pixel_geometry = reference.object_.get_pixel_geometry()
    test_pixel_geometry = test.object_.get_pixel_geometry()

    if reference_pixel_geometry != test_pixel_geometry:
        raise ValueError(
            f'Object pixel geometry mismatch: reference {reference_pixel_geometry} '
            f'vs moving {test_pixel_geometry}!'
        )

    common_height_px = min(reference.object_.height_px, test.object_.height_px)
    common_width_px = min(reference.object_.width_px, test.object_.width_px)
    cropped_reference_object = center_crop_object(
        reference.object_, common_height_px, common_width_px
    )
    cropped_test_object = center_crop_object(test.object_, common_height_px, common_width_px)
    cropped_reference = replace(reference, object_=cropped_reference_object)

    # Bootstrap on a quantity that is blind to the ramp ambiguity still present
    # at this point -- VARIATION is the member flagged `is_ramp_invariant` -- and
    # let the loop below refine with the requested quantity once the ramp has
    # been standardized away.
    shift_y_px, shift_x_px = estimate_object_alignment_shift(
        cropped_reference_object,
        cropped_test_object,
        upsample_factor=upsample_factor,
        registration_quantity=RegistrationQuantity.VARIATION,
    )

    iteration = 0
    is_converged = False

    while True:
        aligned_test = replace(
            test,
            object_=shift_object(cropped_test_object, shift_y_px=shift_y_px, shift_x_px=shift_x_px),
        )
        ambiguities = estimate_reconstruction_ambiguities(
            aligned_test, reference=cropped_reference, weights=weights
        )
        standardized_test = ambiguities.standardize_product(aligned_test)
        iteration += 1

        if is_converged:
            # The residual measured last pass has already been folded into the
            # shift, so the pair standardized just now is the converged one.
            break

        residual_y_px, residual_x_px = estimate_object_alignment_shift(
            cropped_reference_object,
            standardized_test.object_,
            upsample_factor=upsample_factor,
            registration_quantity=registration_quantity,
        )
        residual_px = float(numpy.hypot(residual_y_px, residual_x_px))
        is_converged = residual_px < alignment_tolerance_px

        if not is_converged and iteration >= num_alignment_iterations:
            logger.warning(
                f'Object alignment did not converge in {num_alignment_iterations} iterations;'
                f' residual shift is {residual_px:.3f} px.'
            )
            break

        shift_y_px += residual_y_px
        shift_x_px += residual_x_px

    logger.debug(
        f'Object alignment finished after {iteration} iteration(s) at'
        f' (y, x) = ({shift_y_px:.4f}, {shift_x_px:.4f}) px.'
    )

    reference_array = cropped_reference_object.get_layers_flattened()
    test_array = standardized_test.object_.get_layers_flattened()

    common_dtype = numpy.result_type(reference_array.dtype, test_array.dtype)

    return ObjectComparison(
        reference_complex=reference_array.astype(common_dtype, copy=False),
        test_complex=test_array.astype(common_dtype, copy=False),
        pixel_geometry=cropped_reference_object.get_pixel_geometry().copy(),
        ambiguities=ambiguities,
    )


class ApodizationWindow(Enum):
    """Separable real window multiplied onto an image before its FFT.

    A reconstruction has hard edges: the object array simply stops, and the
    un-illuminated border is not the continuation of the sample. FFTing that
    discontinuity sprays cross-shaped leakage across every ring, which inflates
    the apparent correlation at high frequency. A window tapers the edge away.

    :attr:`TUKEY` is a *flat-cored* taper: the cosine roll-off occupies
    ``taper_fraction`` of each axis, split between the two ends, and the
    interior is left at unit weight. :attr:`HANN` is the ``taper_fraction=1``
    limit of it, which throws away most of the object's area -- its effective
    area is only 3/8 of the array -- so tapering only the outer margin is
    preferred. Increase ``taper_fraction`` until the correlation stops changing.

    :attr:`NONE` scores the array as it stands. That is right only when the
    edges are already soft, either because the caller passed its own window
    array instead or because the data is periodic.
    """

    TUKEY = auto()
    HANN = auto()
    NONE = auto()

    def build(self, shape: tuple[int, int], *, taper_fraction: float) -> RealArrayType | None:
        """Materialize the window, or None when the multiply can be skipped."""
        match self:
            case ApodizationWindow.NONE:
                return None
            case ApodizationWindow.HANN:
                alpha = 1.0
            case ApodizationWindow.TUKEY:
                alpha = float(numpy.clip(taper_fraction, 0.0, 1.0))

                if alpha <= 0.0:
                    return None

        height_px, width_px = shape
        return numpy.multiply.outer(tukey(height_px, alpha), tukey(width_px, alpha))


def _make_apodization_window(
    shape: tuple[int, int],
    window: ApodizationWindow | RealArrayType,
    *,
    taper_fraction: float,
) -> RealArrayType | None:
    """Resolve a window choice, or a caller-supplied window array, to an array.

    Returns ``None`` when the multiply can be skipped entirely.
    """
    if isinstance(window, ApodizationWindow):
        return window.build(shape, taper_fraction=taper_fraction)

    window_arr = numpy.asarray(window, dtype=numpy.float64)

    if window_arr.shape != shape:
        raise ValueError(f'Window shape {window_arr.shape} does not match image shape {shape}!')

    if not numpy.all(numpy.isfinite(window_arr)):
        raise ValueError('Window must be finite!')

    if not numpy.any(window_arr):
        raise ValueError('Window must have at least one nonzero element!')

    return window_arr


def _compute_ring_geometry(
    shape: tuple[int, int],
    pixel_width_m: float,
    pixel_height_m: float,
    num_bins: int | None,
    *,
    exclude_dc: bool = False,
    ring_edge_tolerance: float = 1.0e-9,
) -> tuple[IntegerArrayType, RealArrayType, int, float]:
    """Assign every Fourier pixel inside the inscribed Nyquist circle to a ring.

    The rings tile ``[0, f_nyquist]`` where ``f_nyquist = min(1 / (2 * dy),
    1 / (2 * dx))`` is the largest spatial frequency sampled in *every*
    direction. Fourier pixels beyond that radius -- the corners of the FFT grid,
    and for anisotropic pixels a whole band along the finely sampled axis -- are
    dropped, because a ring that is only partially populated mixes a
    direction-dependent subset of the spectrum into a quantity reported as
    isotropic and corrupts the per-ring pixel count the van Heel/Schatz
    threshold depends on.

    With ``exclude_dc`` the zero-frequency pixel is dropped as well. Removing
    the mean of a windowed image leaves its DC coefficient at round-off rather
    than exactly zero, and a lone-pixel ring normalizes that round-off back up
    to a correlation of order one; dropping the pixel makes the ring honestly
    empty instead.

    Radii are compared against ring edges in units of the bin width, where an
    exact multiple -- the axis Nyquist frequency, say -- is only exact in real
    arithmetic. ``ring_edge_tolerance`` nudges the ratio so a boundary pixel
    lands in the ring it belongs to instead of the one below.

    Returns ``(rings, inside, num_bins, bin_size_per_m)`` where ``inside`` is a
    flat boolean mask over the FFT grid and ``rings`` holds the ring index of
    each selected pixel, in the order ``flat_array[inside]`` produces them.
    """
    height_px, width_px = shape
    kx_per_m = scipy.fft.fftfreq(width_px, d=pixel_width_m)
    ky_per_m = scipy.fft.fftfreq(height_px, d=pixel_height_m)
    nyquist_per_m = min(0.5 / pixel_height_m, 0.5 / pixel_width_m)

    radii_per_m = numpy.hypot(ky_per_m[:, None], kx_per_m[None, :]).ravel()
    inside = radii_per_m <= nyquist_per_m * (1.0 + ring_edge_tolerance)

    if exclude_dc:
        inside[0] = False

    inside_radii_per_m = radii_per_m[inside]

    if num_bins is None:
        step_x_per_m = abs(kx_per_m[1]) if width_px > 1 else 0.0
        step_y_per_m = abs(ky_per_m[1]) if height_px > 1 else 0.0
        bin_size_per_m = max(step_x_per_m, step_y_per_m)

        if bin_size_per_m <= 0.0:
            raise ValueError('Arrays must have at least two pixels along one axis!')

        rings = numpy.floor(inside_radii_per_m / bin_size_per_m + ring_edge_tolerance).astype(
            numpy.intp, copy=False
        )
        resolved_num_bins = int(rings.max()) + 1
    else:
        if num_bins < 1:
            raise ValueError(f'Number of bins must be positive; got {num_bins}!')

        resolved_num_bins = num_bins
        bin_size_per_m = nyquist_per_m / num_bins
        rings = numpy.floor(inside_radii_per_m / bin_size_per_m + ring_edge_tolerance).astype(
            numpy.intp, copy=False
        )
        numpy.clip(rings, 0, num_bins - 1, out=rings)

    return rings, inside, resolved_num_bins, bin_size_per_m


def _prepare_spectral_image(
    array: ComplexArrayType | RealArrayType,
    window: RealArrayType | None,
    *,
    subtract_mean: bool,
) -> numpy.ndarray:
    """Remove the DC term and apodize an image ahead of its FFT.

    The mean is *window-weighted* so that the zeroth Fourier coefficient of the
    returned array is exactly zero. Subtracting the unweighted mean first would
    leave a residual DC term after the taper, and subtracting a constant after
    the taper would reintroduce the hard edge the taper exists to remove.
    """
    values = numpy.asarray(array)

    if subtract_mean:
        if window is None:
            values = values - values.mean()
        else:
            values = values - (values * window).sum() / window.sum()

    return values if window is None else values * window


@dataclass(frozen=True)
class FourierRingCorrelation:
    """Per-ring Fourier ring correlation between two complex images, with resolution estimators."""

    spatial_frequency_per_m: RealArrayType
    correlation: RealArrayType
    pixels_per_ring: IntegerArrayType

    def get_resolution_m(self, threshold: float) -> float:
        threshold_curve = numpy.full(self.correlation.shape, threshold, dtype=float)
        return self._resolution_at_threshold_curve(threshold_curve)

    def get_resolution_m_at_bit_threshold(self, bits: float = 0.5) -> float:
        """Resolution at the van Heel & Schatz b-bit FRC threshold curve.

        See: M. van Heel and M. Schatz, "Fourier shell correlation threshold
        criteria," J. Struct. Biol. 151, 250-262 (2005). ``bits=0.5`` is the
        half-bit criterion; ``bits=1.0`` is the 1-bit criterion. The threshold
        is shaped by the number of Fourier pixels per ring, so noisier
        low-frequency rings tolerate higher correlation before being deemed
        significant.
        """
        return self._resolution_at_threshold_curve(self.get_bit_threshold_curve(bits))

    def get_bit_threshold_curve(self, bits: float = 0.5) -> RealArrayType:
        """Per-ring van Heel/Schatz b-bit FRC significance threshold.

        Bins with zero pixels yield NaN.
        """
        sigma = 0.5 * (2.0**bits - 1.0)
        sqrt_sigma = numpy.sqrt(sigma)
        n_per_ring = numpy.asarray(self.pixels_per_ring, dtype=float)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            inv_sqrt_n = numpy.where(n_per_ring > 0.0, 1.0 / numpy.sqrt(n_per_ring), numpy.nan)
            threshold_curve = (sigma + (2.0 * sqrt_sigma + 1.0) * inv_sqrt_n) / (
                sigma + 1.0 + 2.0 * sqrt_sigma * inv_sqrt_n
            )

        return threshold_curve

    def get_spectral_signal_to_noise_ratio(self) -> RealArrayType:
        """Spectral SNR per ring under the full-image van Heel/Schatz convention.

        ``SSNR(f) = 2 * FRC(f) / (1 - FRC(f))``. Negative FRC values (anti-
        correlation from noise, not physical signal) are clipped to 0 so SSNR=0.
        FRC values of exactly 1 yield +inf. NaN inputs propagate as NaN.
        """
        frc = numpy.asarray(self.correlation, dtype=float)
        nan_mask = numpy.isnan(frc)
        safe_frc = numpy.clip(frc, 0.0, None)
        denominator = 1.0 - safe_frc

        with numpy.errstate(divide='ignore', invalid='ignore'):
            ssnr = numpy.where(denominator > 0.0, 2.0 * safe_frc / denominator, numpy.inf)

        return numpy.where(nan_mask, numpy.nan, ssnr)

    def get_area_under_curve(
        self,
        *,
        normalize: bool = True,
        max_frequency_per_m: float | None = None,
    ) -> float:
        """Trapezoidal area under the FRC curve over spatial frequency.

        NaN correlation bins are excluded. With ``normalize=True`` the integral
        is divided by the span of the integration domain, giving a dimensionless
        number in [0, 1] (1 = ideal FRC, 0 = uncorrelated). With
        ``normalize=False`` the result has units of m^-1. ``max_frequency_per_m``
        optionally clips the upper end of the integration domain.
        """
        freq = numpy.asarray(self.spatial_frequency_per_m, dtype=float)
        corr = numpy.asarray(self.correlation, dtype=float)

        mask = numpy.isfinite(corr) & numpy.isfinite(freq)
        if max_frequency_per_m is not None:
            mask &= freq <= max_frequency_per_m

        if mask.sum() < 2:
            return float('nan')

        f = freq[mask]
        y = corr[mask]
        span = f[-1] - f[0]

        if span <= 0.0:
            return float('nan')

        auc = numpy.trapezoid(y, f)
        return float(auc / span if normalize else auc)

    def get_average_signal_to_noise_ratio(self) -> float:
        """Mean SSNR across bins, excluding non-finite values (NaN and +inf)."""
        ssnr = self.get_spectral_signal_to_noise_ratio()
        finite = numpy.isfinite(ssnr)
        if not finite.any():
            return float('nan')

        return float(numpy.mean(ssnr[finite]))

    def get_resolution_m_at_signal_to_noise_threshold(self, snr: float) -> float:
        """Resolution where the FRC-derived SSNR drops below ``snr``.

        Inverts ``SSNR = 2 * FRC / (1 - FRC)`` to get the equivalent FRC
        threshold ``F = snr / (snr + 2)`` and defers to
        :meth:`get_resolution_m`. Raises ``ValueError`` for negative ``snr``.
        """
        if snr < 0.0:
            raise ValueError('SNR threshold must be non-negative')
        frc_threshold = snr / (snr + 2.0)
        return self.get_resolution_m(frc_threshold)

    def _resolution_at_threshold_curve(self, threshold_curve: RealArrayType) -> float:
        freq = numpy.asarray(self.spatial_frequency_per_m)
        diff = numpy.asarray(self.correlation) - threshold_curve

        finite = numpy.isfinite(diff)
        below = numpy.flatnonzero(finite & (diff < 0.0))
        if below.size == 0:
            return float('nan')

        first = below[0]
        # Only interpolate when the previous bin was strictly above the
        # threshold. If it merely touched the threshold (diff == 0), the
        # crossing belongs at freq[first], not at the touchpoint. The DC bin
        # used to be the case in point -- FRC == 1 against the bit-threshold's
        # N == 1 limit of 1 -- but under the default mean subtraction that bin
        # is empty and its correlation is NaN, so the `finite` mask drops it.
        if first > 0 and finite[first - 1] and diff[first - 1] > 0.0:
            g0 = diff[first - 1]
            g1 = diff[first]
            alpha = g0 / (g0 - g1)
            crossing_freq = freq[first - 1] + alpha * (freq[first] - freq[first - 1])
        else:
            crossing_freq = freq[first]

        return float('nan') if crossing_freq <= 0.0 else float(1.0 / crossing_freq)


def compute_fourier_ring_correlation(
    array1: ComplexArrayType,
    array2: ComplexArrayType,
    pixel_width_m: float,
    pixel_height_m: float,
    *,
    window: ApodizationWindow | RealArrayType = ApodizationWindow.TUKEY,
    taper_fraction: float = 0.2,
    subtract_mean: bool = True,
    num_bins: int | None = None,
    workers: int = -1,
) -> FourierRingCorrelation:
    """Compute the Fourier ring correlation between two complex images.

    See: Joan Vila-Comamala, Ana Diaz, Manuel Guizar-Sicairos, Alexandre Mantion,
    Cameron M. Kewish, Andreas Menzel, Oliver Bunk, and Christian David,
    "Characterization of high-resolution diffractive X-ray optics by ptychographic
    coherent diffractive imaging," Opt. Express 19, 21333-21344 (2011)

    The ring statistic is the signed real part ``sum Re(F1 conj(F2))`` over each
    ring, normalized by ``sqrt(sum |F1|^2 sum |F2|^2)``. That is the convention
    the van Heel/Schatz threshold curve in
    :meth:`FourierRingCorrelation.get_bit_threshold_curve` was derived for: it
    fluctuates about zero on uncorrelated rings, whereas the modulus
    ``|sum F1 conj(F2)|`` -- also in common use, and more forgiving of an
    imperfectly removed phase ambiguity -- has a positive bias of roughly
    ``0.886 / sqrt(N)`` there and so reports optimistic resolution. The two
    agree wherever the ambiguities have genuinely been removed, which is what
    :func:`compute_object_comparison` is for.

    Args:
        array1: Reference image, 2D.
        array2: Test image, same shape as ``array1``.
        pixel_width_m: Real-space pixel width.
        pixel_height_m: Real-space pixel height.
        window: Apodization applied before the FFT: an
            :class:`ApodizationWindow` member, or a custom 2D array of the same
            shape -- pass an illumination-derived soft mask here, e.g.
            :attr:`ScoringRegion.weights`.
        taper_fraction: Fraction of each axis occupied by the Tukey cosine
            roll-off, split between the two ends.
        subtract_mean: Remove each image's complex mean before the FFT. A
            ptychography object is ``O ~ 1 + small``, so the DC bin is dominated
            by the background rather than by structure; with it removed the DC
            ring becomes ``0/0`` and is reported as NaN.
        num_bins: Number of rings tiling ``[0, f_nyquist]``. ``None`` (default)
            keeps the natural FFT bin width ``max(1 / (W dx), 1 / (H dy))``.
        workers: Passed to ``scipy.fft.fft2``.

    Returns:
        A :class:`FourierRingCorrelation` whose ``spatial_frequency_per_m``
        holds ring *left edges*.
    """
    if array1.ndim != 2 or array2.ndim != 2:
        raise ValueError('Arrays must be 2D!')

    if array1.shape != array2.shape:
        raise ValueError(f'Arrays must have same shape; got {array1.shape} vs {array2.shape}!')

    shape = (array1.shape[0], array1.shape[1])
    apodization = _make_apodization_window(shape, window, taper_fraction=taper_fraction)
    values1 = _prepare_spectral_image(array1, apodization, subtract_mean=subtract_mean)
    values2 = _prepare_spectral_image(array2, apodization, subtract_mean=subtract_mean)

    rings, inside, resolved_num_bins, bin_size_per_m = _compute_ring_geometry(
        shape, pixel_width_m, pixel_height_m, num_bins, exclude_dc=subtract_mean
    )

    sf1 = scipy.fft.fft2(values1, workers=workers).ravel()[inside]
    sf2 = scipy.fft.fft2(values2, workers=workers).ravel()[inside]

    # |F|^2 as real weights — avoids complex bincount and the imaginary residue
    # of F * conj(F).
    power1 = sf1.real * sf1.real + sf1.imag * sf1.imag
    power2 = sf2.real * sf2.real + sf2.imag * sf2.imag
    cross_real = sf1.real * sf2.real + sf1.imag * sf2.imag

    c11 = numpy.bincount(rings, weights=power1, minlength=resolved_num_bins)
    c22 = numpy.bincount(rings, weights=power2, minlength=resolved_num_bins)
    c12 = numpy.bincount(rings, weights=cross_real, minlength=resolved_num_bins)
    pixels_per_ring = numpy.bincount(rings, minlength=resolved_num_bins)

    denominator = numpy.sqrt(c11 * c22)

    with numpy.errstate(invalid='ignore', divide='ignore'):
        correlation = numpy.where(denominator > 0.0, c12 / denominator, numpy.nan)

    return FourierRingCorrelation(
        spatial_frequency_per_m=numpy.arange(resolved_num_bins) * bin_size_per_m,
        correlation=numpy.clip(correlation, -1.0, 1.0),
        pixels_per_ring=pixels_per_ring,
    )


@dataclass(frozen=True)
class PowerSpectralDensity:
    """Radially averaged power spectral density of a single reconstruction."""

    spatial_frequency_per_m: RealArrayType
    """Ring left edges, matching :class:`FourierRingCorrelation`."""
    power_spectral_density_m2: RealArrayType
    """Ring mean of the normalized ``|F|^2``; NaN for empty rings."""
    pixels_per_ring: IntegerArrayType


def compute_power_spectral_density(
    array: ComplexArrayType | RealArrayType,
    pixel_width_m: float,
    pixel_height_m: float,
    *,
    window: ApodizationWindow | RealArrayType = ApodizationWindow.TUKEY,
    taper_fraction: float = 0.2,
    subtract_mean: bool = True,
    num_bins: int | None = None,
    workers: int = -1,
) -> PowerSpectralDensity:
    """Radially averaged power spectral density of one image.

    Unlike the FRC this needs only a single reconstruction, so it says something
    about a product compared with itself: where the spectrum rolls off into a
    flat noise floor is a direct readout of the reconstruction's information
    content.

    The 2D density is ``dx dy |F|^2 / (N mean(w^2))``. Dividing by the mean
    squared window compensates the power the apodization removes, so the
    reported level does not depend on the window choice. Because nothing is
    zero-padded, Parseval holds exactly: integrating the unwindowed density over
    spatial frequency recovers ``mean(|array - mean(array)|^2)``. Units are the
    input units squared times m^2; a reconstructed object is a dimensionless
    complex transmission, so the result is in m^2.

    Rings are the arithmetic *mean* over the ring, not the sum the FRC
    accumulates, so the curve is independent of how many Fourier pixels a ring
    happens to contain.
    """
    if array.ndim != 2:
        raise ValueError('Array must be 2D!')

    shape = (array.shape[0], array.shape[1])
    apodization = _make_apodization_window(shape, window, taper_fraction=taper_fraction)
    values = _prepare_spectral_image(array, apodization, subtract_mean=subtract_mean)

    rings, inside, resolved_num_bins, bin_size_per_m = _compute_ring_geometry(
        shape, pixel_width_m, pixel_height_m, num_bins, exclude_dc=subtract_mean
    )

    window_power = 1.0 if apodization is None else numpy.mean(numpy.square(apodization))
    scale = pixel_width_m * pixel_height_m / (values.size * window_power)

    sf = scipy.fft.fft2(values, workers=workers).ravel()[inside]
    power = scale * (sf.real * sf.real + sf.imag * sf.imag)

    ring_sum = numpy.bincount(rings, weights=power, minlength=resolved_num_bins)
    pixels_per_ring = numpy.bincount(rings, minlength=resolved_num_bins)

    with numpy.errstate(invalid='ignore', divide='ignore'):
        ring_mean = numpy.where(pixels_per_ring > 0, ring_sum / pixels_per_ring, numpy.nan)

    return PowerSpectralDensity(
        spatial_frequency_per_m=numpy.arange(resolved_num_bins) * bin_size_per_m,
        power_spectral_density_m2=ring_mean,
        pixels_per_ring=pixels_per_ring,
    )


@dataclass(frozen=True)
class ScoringRegion:
    """A rectangular, tapered sub-window of an object array to score metrics over.

    ``weights`` covers only the rectangle ``[row_begin:row_end,
    column_begin:column_end]`` of an array of shape ``array_shape``. Use
    :meth:`crop` to cut a matching array down to that rectangle; the cropped
    array and ``weights`` are then the matching pair to hand to
    :func:`compute_fourier_ring_correlation` or
    :func:`compute_power_spectral_density`.
    """

    array_shape: tuple[int, int]
    row_begin: int
    row_end: int
    column_begin: int
    column_end: int
    weights: RealArrayType

    @property
    def height_px(self) -> int:
        return self.row_end - self.row_begin

    @property
    def width_px(self) -> int:
        return self.column_end - self.column_begin

    def crop(self, array: ComplexArrayType | RealArrayType) -> numpy.ndarray:
        """Cut the trailing two axes of ``array`` down to this region's rectangle."""
        values = numpy.asarray(array)

        if values.shape[-2:] != self.array_shape:
            raise ValueError(
                f'Arrays must have same shape; got {values.shape[-2:]} vs {self.array_shape}!'
            )

        return values[..., self.row_begin : self.row_end, self.column_begin : self.column_end]


def _largest_true_rectangle(mask: numpy.ndarray) -> tuple[int, int, int, int]:
    """Find the largest axis-aligned all-true rectangle in a 2D boolean mask.

    Returns ``(row_begin, row_end, column_begin, column_end)``; an all-false
    mask yields an empty rectangle at the origin. This is the classic
    O(height * width) "largest rectangle in a histogram" sweep: each row keeps
    a running column-height histogram of consecutive true pixels ending at that
    row, and a monotonic stack finds the widest span each height can cover. A
    sentinel zero is appended to every histogram so the stack always drains.
    """
    grid = numpy.asarray(mask, dtype=bool)
    height_px, width_px = grid.shape
    heights = [0] * (width_px + 1)
    best_area = 0
    best = (0, 0, 0, 0)

    for row in range(height_px):
        for column in range(width_px):
            heights[column] = heights[column] + 1 if grid[row, column] else 0

        stack: list[int] = []

        for column in range(width_px + 1):
            while stack and heights[stack[-1]] >= heights[column]:
                bar = stack.pop()
                span_begin = stack[-1] + 1 if stack else 0
                area = heights[bar] * (column - span_begin)

                if area > best_area:
                    best_area = area
                    best = (row + 1 - heights[bar], row + 1, span_begin, column)

            stack.append(column)

    return best


def compute_illumination_scoring_region(
    illumination_map: IlluminationMap,
    array_shape: tuple[int, int],
    *,
    threshold_fraction: float = 0.5,
    erosion_px: int = 0,
    smoothing_px: float = 0.0,
    taper_fraction: float = 0.2,
    min_region_px: int = 8,
) -> ScoringRegion:
    """Derive a rectangular, tapered scoring window from where the probe actually was.

    Object pixels outside the scan are not reconstructions of anything: they
    hold whatever the initializer left there. Scoring them drags every metric
    toward the initializer and, because the illuminated patch has a hard edge,
    injects the very leakage the apodization window exists to suppress.
    Restricting the score to an illumination-derived region of interest avoids
    both.

    The mask is ``photon_number > threshold_fraction * mean(photon_number)``
    over the illuminated pixels, optionally smoothed and eroded to pull the
    boundary in off the ragged edge of the outermost probe positions, and then
    reduced to its largest inscribed rectangle so the result is still FFT-able.
    The rectangle is finally multiplied by a Tukey taper.

    Args:
        illumination_map: Photon-count canvas for the product being scored.
        array_shape: Shape of the object array the region will be applied to.
            The illumination map is center-cropped to it.
        threshold_fraction: Illuminated-pixel mean fraction above which a pixel
            counts as illuminated.
        erosion_px: Binary erosion radius, in pixels, applied to the mask. A
            useful value is a fraction of the probe width, which pulls the
            boundary inside the half-illuminated rim left by the outermost scan
            positions.
        smoothing_px: Gaussian smoothing sigma applied to the photon counts
            before thresholding, to keep scan-grid ripple from punching holes
            in the mask.
        taper_fraction: Tukey taper fraction for the returned weights.
        min_region_px: Smallest rectangle edge, in pixels, worth returning.
            Below this an FFT has too few rings to say anything, and the
            caller is better off scoring the whole array.

    Raises:
        ValueError: If the illumination map is smaller than ``array_shape``,
            carries no photons, or leaves a rectangle too small to carry a
            meaningful spectrum. Callers are expected to catch the last case
            and fall back to scoring the whole array.
    """
    photon_number = numpy.asarray(illumination_map.photon_number, dtype=float)
    target_height_px, target_width_px = array_shape
    delta_height_px = photon_number.shape[0] - target_height_px
    delta_width_px = photon_number.shape[1] - target_width_px

    if delta_height_px < 0 or delta_width_px < 0:
        raise ValueError(
            f'Illumination map shape {photon_number.shape} is smaller than '
            f'object shape {array_shape}!'
        )

    row_offset = delta_height_px // 2
    column_offset = delta_width_px // 2
    photon_number = photon_number[
        row_offset : row_offset + target_height_px,
        column_offset : column_offset + target_width_px,
    ]

    if smoothing_px > 0.0:
        photon_number = gaussian_filter(photon_number, smoothing_px)

    illuminated = photon_number > 0.0

    if not illuminated.any():
        raise ValueError('Illumination map carries no photons!')

    mask = photon_number > threshold_fraction * photon_number[illuminated].mean()

    if erosion_px > 0:
        size = 2 * erosion_px + 1
        mask = binary_erosion(mask, structure=numpy.ones((size, size), dtype=bool))

    row_begin, row_end, column_begin, column_end = _largest_true_rectangle(mask)

    if row_end - row_begin < min_region_px or column_end - column_begin < min_region_px:
        raise ValueError(
            f'Illumination scoring region is degenerate; threshold fraction '
            f'{threshold_fraction} with erosion {erosion_px} px left a '
            f'{row_end - row_begin} x {column_end - column_begin} rectangle!'
        )

    weights = _make_apodization_window(
        (row_end - row_begin, column_end - column_begin),
        ApodizationWindow.TUKEY,
        taper_fraction=taper_fraction,
    )

    if weights is None:
        weights = numpy.ones((row_end - row_begin, column_end - column_begin), dtype=float)

    return ScoringRegion(
        array_shape=(target_height_px, target_width_px),
        row_begin=row_begin,
        row_end=row_end,
        column_begin=column_begin,
        column_end=column_end,
        weights=weights,
    )


def compute_root_mean_square_error(
    reference: ComplexArrayType | RealArrayType,
    test: ComplexArrayType | RealArrayType,
) -> float:
    """L2-norm pixelwise distance: ``sqrt(mean(|test - reference|**2))``.

    Accepts real or complex inputs. For complex inputs, ``|test - reference|``
    is the modulus of the per-pixel complex difference (Euclidean distance in
    the complex plane), which is the natural error metric for ptychography
    object reconstructions.
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    diff = test - reference
    return float(numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(diff)))))


def compute_mean_absolute_error(
    reference: ComplexArrayType | RealArrayType,
    test: ComplexArrayType | RealArrayType,
) -> float:
    """L1-norm pixelwise distance: ``mean(|test - reference|)``.

    For complex inputs, ``|test - reference|`` is the modulus of the per-pixel
    complex difference. Less sensitive to outliers than
    :func:`compute_root_mean_square_error`.
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    return float(numpy.mean(numpy.absolute(test - reference)))


def compute_r_factor(
    reference: ComplexArrayType | RealArrayType,
    test: ComplexArrayType | RealArrayType,
) -> float:
    """Relative L1 distance: ``sum(|test - reference|) / sum(|reference|)``.

    Unitless, scale-invariant counterpart to
    :func:`compute_mean_absolute_error`. Returns 0 for a perfect match and
    grows without an upper bound as the reconstructions disagree (a fully
    uncorrelated test typically lands near 1). Accepts real or complex
    inputs; for complex inputs both the numerator and denominator use the
    per-pixel modulus, matching the convention of the other metrics in this
    module. Returns NaN when the reference has zero total amplitude (R-factor
    is undefined in that case).
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    denominator = numpy.sum(numpy.absolute(reference))
    if denominator == 0.0:
        return float('nan')

    numerator = numpy.sum(numpy.absolute(test - reference))
    return float(numerator / denominator)


def _infer_data_range(reference: RealArrayType) -> float:
    """Pick a sensible default ``data_range`` for PSNR/SSIM from the reference image.

    Uses ``reference.max() - reference.min()``, matching scikit-image's
    recommendation for floating-point inputs.
    """
    return float(numpy.ptp(reference))


def compute_peak_signal_to_noise_ratio(
    reference: RealArrayType,
    test: RealArrayType,
    *,
    data_range: float | None = None,
) -> float:
    """Peak signal-to-noise ratio in dB via :func:`skimage.metrics.peak_signal_noise_ratio`.

    Real-valued inputs only. When ``data_range`` is ``None``, infers
    ``reference.max() - reference.min()`` so floating-point inputs do not
    trigger scikit-image's data-range warning. Returns ``+inf`` when the two
    arrays are identical.
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    effective_range = _infer_data_range(reference) if data_range is None else data_range
    return float(peak_signal_noise_ratio(reference, test, data_range=effective_range))


def compute_structural_similarity(
    reference: RealArrayType,
    test: RealArrayType,
    *,
    data_range: float | None = None,
) -> float:
    """Structural similarity index via :func:`skimage.metrics.structural_similarity`.

    Real-valued inputs only. When ``data_range`` is ``None``, infers
    ``reference.max() - reference.min()``. Returns a scalar in ``[-1, 1]``;
    ``1.0`` for identical inputs.
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    effective_range = _infer_data_range(reference) if data_range is None else data_range
    return float(structural_similarity(reference, test, data_range=effective_range))


def compute_normalized_mutual_information(
    reference: RealArrayType,
    test: RealArrayType,
    *,
    bins: int = 100,
) -> float:
    """Normalized mutual information via :func:`skimage.metrics.normalized_mutual_information`.

    Real-valued inputs only. Returns the Studholme NMI
    ``(H(reference) + H(test)) / H(reference, test)``, which is ~1.0 for
    statistically independent inputs and 2.0 for identical inputs. Unlike
    SSIM/PSNR, NMI is insensitive to monotonic intensity remappings, so it
    scores residual amplitude scale ambiguity less harshly. ``bins`` controls
    the joint-histogram resolution; the scikit-image default of 100 is
    preserved.
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    return float(normalized_mutual_information(reference, test, bins=bins))


@dataclass(frozen=True)
class ReconstructionResiduals:
    """Real- and reciprocal-space residual maps comparing measured to forward-simulated patterns.

    Both maps are dimensionless amplitude R-factors (Crowther/Rosenthal): the fraction of
    detected amplitude the model fails to explain, with both numerator and denominator scaling
    with local photon count so the ratio decouples from sample thickness, probe brightness, and
    incident flux. A perfectly fitted reconstruction yields zero everywhere; an uncorrelated
    model approaches ~1. The square-root transform inside the metric is Poisson
    variance-stabilizing, so the shot-noise floor of ``R_F`` is automatically tighter where
    photons are abundant and looser where they are scarce — the eye-readable behavior.

    **What "amplitude" means here.** The metric compares **diffraction amplitudes** on the
    detector (``√I_meas``, ``√I_pred``), not **object amplitudes** (``|O|``). The reconstructed
    object is a complex transmission function whose phase and amplitude jointly determine the
    predicted diffraction; phase-dominated samples (typical at hard-x-ray energies) still
    produce richly structured diffraction patterns, and errors in reconstructed phase show up
    as errors in predicted intensity. These maps therefore quantify detector-domain data-fit
    quality and are *not* phase-blind.

    """

    real_space_error_map: RealArrayType
    """2D amplitude R-factor on the object grid.

    NaN where the R-factor is undefined: un-illuminated pixels (no frame contributed) and
    object regions touched only by frames with zero measured signal (``Σ √I_meas = 0``).
    """
    object_pixel_geometry: PixelGeometry
    object_center: ObjectCenter
    reciprocal_space_error_map: RealArrayType
    """2D amplitude R-factor on the detector grid.

    Each pixel is ``Σ_n |√I_meas,n − √I_pred,n| / Σ_n √I_meas,n``, summed across frames. NaN
    at bad pixels and at detector pixels with no measured signal across any frame.
    """
    detector_pixel_geometry: PixelGeometry


def compute_reconstruction_residuals(
    product: Product,
    measured_patterns: DiffractionPatterns,
    bad_pixels: BadPixels,
) -> ReconstructionResiduals:
    """Compute amplitude R-factor real- and reciprocal-space residual maps for a reconstructed product.

    Re-runs the multislice forward model on ``product`` (via
    :func:`generate_diffraction_data`) and compares the simulated intensities
    to ``measured_patterns`` through the Crowther/Rosenthal amplitude R-factor:
    ``Σ |√I_meas − √I_pred| / Σ √I_meas``. The √-transform is the standard
    Poisson variance stabilizer, and the ratio form scales numerator and
    denominator together so brightness and thickness cancel — only model
    misfit moves the value.

    The reciprocal-space map sums numerator and denominator over all frames at
    each detector pixel; bad pixels become NaN. The real-space map aggregates
    per-frame amplitude residual sums and per-frame measured-amplitude sums
    onto the object grid, both weighted by the same per-frame-normalized
    probe-intensity patch (each frame's ``|probe|²`` divided by its own total),
    then divides. The shared probe-weighting makes the ratio scan-density
    invariant (variable-probe frames included) and ensures every frame
    contributes equal total weight regardless of probe power. Un-illuminated
    pixels and detector pixels with no measured signal across any frame both
    remain zero.

    Inputs must already be aligned: ``measured_patterns`` is shape ``(N, H, W)``
    in product position order (typically the output of
    :func:`ptychodus.api.reconstruct.prepare_reconstruct_input`).

    **Hard-x-ray caveat.** At hard-x-ray energies the detector dynamic range commonly spans
    4–6 orders of magnitude and the ``√I`` transform compresses that only to ~2–3 orders, so
    both the numerator and denominator of ``R_F`` are dominated by the bright low-q region.
    The high-q tail — where fine phase-contrast features leave their strongest unique
    signature — is correspondingly underweighted, and a reconstruction with poor high-q fit
    can read a deceptively small ``R_F``. ``bad_pixels`` is the supported lever for masking
    this region: it already drops beamstop pixels, and users who care about high-q phase
    fidelity should extend it to cover the bright direct-beam halo just outside the beamstop.
    Soft-x-ray data has a much smaller detector dynamic range and is not affected to the
    same degree.
    """
    if measured_patterns.ndim != 3:
        raise ValueError(
            f'measured_patterns must be 3D (N,H,W); got shape {measured_patterns.shape}'
        )

    if measured_patterns.shape[1:] != bad_pixels.shape:
        raise ValueError(
            'measured_patterns frame shape does not match bad_pixels shape '
            f'(measured frame={measured_patterns.shape[1:]} vs bad_pixels={bad_pixels.shape})'
        )

    simulated = generate_diffraction_data(product)
    predicted = simulated.get_patterns()

    if predicted.shape != measured_patterns.shape:
        raise ValueError(
            'Simulated patterns shape does not match measured patterns shape '
            f'(simulated={predicted.shape} vs measured={measured_patterns.shape})'
        )

    valid = numpy.logical_not(bad_pixels)
    # Amplitude (sqrt-intensity) form: variance-stabilizes Poisson noise and
    # gives the R-factor a natural unbounded-positive denominator without any
    # ad-hoc clip on small predicted values.
    meas_amp = numpy.sqrt(numpy.maximum(measured_patterns, 0.0))  # (N, H, W)
    pred_amp = numpy.sqrt(numpy.maximum(predicted, 0.0))  # (N, H, W)
    abs_amp_diff = numpy.absolute(meas_amp - pred_amp)  # (N, H, W)

    numerator_per_pixel = abs_amp_diff.sum(axis=0)  # (H, W)
    denominator_per_pixel = meas_amp.sum(axis=0)  # (H, W)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        recip_ratio = numpy.where(
            denominator_per_pixel > 0.0,
            numerator_per_pixel / denominator_per_pixel,
            numpy.nan,
        )
    reciprocal_map = numpy.where(valid, recip_ratio, numpy.nan)

    valid_f = valid.astype(abs_amp_diff.dtype)
    per_frame_numerator = numpy.einsum('nhw,hw->n', abs_amp_diff, valid_f)  # (N,)
    per_frame_denominator = numpy.einsum('nhw,hw->n', meas_amp, valid_f)  # (N,)

    object_geometry = product.object_.get_geometry()
    probe_geometry = product.probes.get_geometry()
    numerator_splat = numpy.zeros((object_geometry.height_px, object_geometry.width_px))
    denominator_splat = numpy.zeros_like(numerator_splat)

    for num_i, den_i, (scan_point, probe) in zip(
        per_frame_numerator, per_frame_denominator, product.iter_position_probes()
    ):
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        bounds = probe_geometry.resolve_patch_bounds(object_point.x_px, object_point.y_px)

        shifted_modes = fourier_shift_2d(probe.get_array(), dx=bounds.dx, dy=bounds.dy)
        intensity = numpy.sum(numpy.abs(shifted_modes) ** 2, axis=0)
        total = intensity.sum()
        patch = intensity / total if total > 0.0 else intensity

        numerator_splat[bounds.y_slice, bounds.x_slice] += num_i * patch
        denominator_splat[bounds.y_slice, bounds.x_slice] += den_i * patch

    real_space_error_map = numpy.full_like(numerator_splat, numpy.nan)
    numpy.divide(
        numerator_splat,
        denominator_splat,
        out=real_space_error_map,
        where=denominator_splat > 0.0,
    )

    return ReconstructionResiduals(
        real_space_error_map=real_space_error_map,
        object_pixel_geometry=object_geometry.get_pixel_geometry(),
        object_center=object_geometry.get_center(),
        reciprocal_space_error_map=reciprocal_map,
        detector_pixel_geometry=simulated.get_pixel_geometry(),
    )
