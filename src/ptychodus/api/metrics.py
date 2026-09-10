"""Resolution metrics for reconstructed images (Fourier ring correlation, etc.)."""

from __future__ import annotations
from dataclasses import dataclass, replace

import numpy
import scipy.fft
from skimage.metrics import (
    normalized_mutual_information,
    peak_signal_noise_ratio,
    structural_similarity,
)

from .typing import ComplexArrayType, IntegerArrayType, RealArrayType
from .diffraction import BadPixels, DiffractionPatterns
from .fourier import fourier_shift_2d
from .simulate.diffraction import generate_diffraction_data
from .geometry import PixelGeometry
from .object import ObjectCenter, align_objects
from .product import Product
from .reconstruct import ReconstructionAmbiguities


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
) -> ObjectComparison:
    """Align ``test`` onto ``reference``, standardize ambiguities, and bundle the pair.

    Args:
        reference: The reconstruction treated as ground truth. Defines the
            array indexing and ambiguity anchor.
        test: The reconstruction being evaluated against ``reference``.
        upsample_factor: Sub-pixel precision for the phase-cross-correlation
            registration.
        weights: Optional non-negative per-pixel weights for the ambiguity
            estimate, shape ``(height_px, width_px)`` matching layer 0. Pass
            a 0/1 mask to restrict the estimate to a region of interest.

    Raises:
        ValueError: If the two products' objects disagree on pixel geometry or
            if the weighted reference intensity is zero.
    """
    cropped_reference_object, aligned_test_object = align_objects(
        reference.object_, test.object_, upsample_factor=upsample_factor
    )
    cropped_reference = replace(reference, object_=cropped_reference_object)
    aligned_test = replace(test, object_=aligned_test_object)

    ambiguities = estimate_reconstruction_ambiguities(
        aligned_test, reference=cropped_reference, weights=weights
    )
    standardized_test = ambiguities.standardize_product(aligned_test)

    reference_array = cropped_reference_object.get_layers_flattened()
    test_array = standardized_test.object_.get_layers_flattened()

    common_dtype = numpy.result_type(reference_array.dtype, test_array.dtype)

    return ObjectComparison(
        reference_complex=reference_array.astype(common_dtype, copy=False),
        test_complex=test_array.astype(common_dtype, copy=False),
        pixel_geometry=cropped_reference_object.get_pixel_geometry().copy(),
        ambiguities=ambiguities,
    )


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
        # Only interpolate when the previous bin was strictly above the threshold.
        # If it merely touched the threshold (diff == 0, e.g. the DC bin where
        # FRC == 1 against the bit-threshold's N=1 limit of 1), the crossing
        # belongs at freq[first], not at the touchpoint.
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
    workers: int = -1,
) -> FourierRingCorrelation:
    """Compute the Fourier ring correlation between two complex images.

    See: Joan Vila-Comamala, Ana Diaz, Manuel Guizar-Sicairos, Alexandre Mantion,
    Cameron M. Kewish, Andreas Menzel, Oliver Bunk, and Christian David,
    "Characterization of high-resolution diffractive X-ray optics by ptychographic
    coherent diffractive imaging," Opt. Express 19, 21333-21344 (2011)
    """
    if array1.ndim != 2 or array2.ndim != 2:
        raise ValueError('Arrays must be 2D!')

    if array1.shape != array2.shape:
        raise ValueError('Arrays must have same shape!')

    height_px, width_px = array1.shape

    kx_per_m = scipy.fft.fftfreq(width_px, d=pixel_width_m)
    ky_per_m = scipy.fft.fftfreq(height_px, d=pixel_height_m)
    bin_size_per_m = max(abs(kx_per_m[1]), abs(ky_per_m[1]))

    radii_per_m = numpy.hypot(ky_per_m[:, None], kx_per_m[None, :])
    rings = numpy.floor(radii_per_m / bin_size_per_m).astype(numpy.intp, copy=False)
    flat_rings = rings.ravel()
    n_bins = int(flat_rings.max()) + 1

    sf1 = scipy.fft.fft2(array1, workers=workers)
    sf2 = scipy.fft.fft2(array2, workers=workers)

    # |F|^2 as real weights — avoids complex bincount and the imaginary residue
    # of F * conj(F).
    power1 = (sf1.real * sf1.real + sf1.imag * sf1.imag).ravel()
    power2 = (sf2.real * sf2.real + sf2.imag * sf2.imag).ravel()
    cross = (sf1 * sf2.conj()).ravel()

    c11 = numpy.bincount(flat_rings, weights=power1, minlength=n_bins)
    c22 = numpy.bincount(flat_rings, weights=power2, minlength=n_bins)
    c12_re = numpy.bincount(flat_rings, weights=cross.real, minlength=n_bins)
    c12_im = numpy.bincount(flat_rings, weights=cross.imag, minlength=n_bins)
    pixels_per_ring = numpy.bincount(flat_rings, minlength=n_bins)

    numerator = numpy.hypot(c12_re, c12_im)
    denominator = numpy.sqrt(c11 * c22)

    with numpy.errstate(invalid='ignore', divide='ignore'):
        correlation = numpy.where(denominator > 0.0, numerator / denominator, numpy.nan)

    rnyquist = min(array1.shape) // 2 + 1
    freqs_per_m = numpy.arange(n_bins) * bin_size_per_m

    return FourierRingCorrelation(
        spatial_frequency_per_m=freqs_per_m[:rnyquist],
        correlation=correlation[:rnyquist],
        pixels_per_ring=pixels_per_ring[:rnyquist],
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
