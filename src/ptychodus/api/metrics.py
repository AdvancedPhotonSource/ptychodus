"""Resolution metrics for reconstructed images (Fourier ring correlation, etc.)."""

from __future__ import annotations
from dataclasses import dataclass, replace

import numpy
import scipy.fft
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .common import ComplexArrayType, IntegerArrayType, RealArrayType
from .geometry import PixelGeometry
from .object import align_objects
from .product import Product
from .reconstructor import ReconstructionAmbiguities


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

    Construct via :meth:`from_products`, which:

    1. Sub-pixel registers the test object onto the reference (:func:`align_objects`).
    2. Estimates the four ambiguity scalars and standardizes the test product
       (:class:`ReconstructionAmbiguities`).
    3. Flattens multi-layer objects via :meth:`Object.get_layers_flattened`.
    4. Promotes both arrays to a common complex dtype.

    Attributes:
        reference_complex: 2D complex array, the reference object's flattened layers,
            promoted to the common dtype.
        test_complex: 2D complex array, the standardized + aligned test object's
            flattened layers. Same shape and dtype as ``reference_complex``.
        pixel_geometry: Shared pixel geometry of both reconstructions (validated
            equal by the upstream primitives).
        ambiguities: The ambiguities removed from the test side, useful as
            provenance (e.g. for reporting "how much ramp/scale was removed"
            alongside the metric value).
    """

    reference_complex: ComplexArrayType
    test_complex: ComplexArrayType
    pixel_geometry: PixelGeometry
    ambiguities: ReconstructionAmbiguities

    @classmethod
    def from_products(
        cls,
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
            upsample_factor: Sub-pixel precision for
                :func:`skimage.registration.phase_cross_correlation`, forwarded
                through :func:`align_objects`.
            weights: Optional non-negative per-pixel weights for the ambiguity
                estimate, shape ``(height_px, width_px)`` matching layer 0. Pass
                a 0/1 mask to restrict the estimate to a region of interest.

        Raises:
            ValueError: If the two products' objects disagree on pixel geometry,
                if their flattened or layer-0 shapes differ, or if the weighted
                reference intensity is zero. These checks come from
                :func:`align_objects` and
                :meth:`ReconstructionAmbiguities.estimate`; this method does not
                duplicate them.
        """
        aligned_test_object = align_objects(
            reference.object_, test.object_, upsample_factor=upsample_factor
        )
        aligned_test = replace(test, object_=aligned_test_object)

        ambiguities = ReconstructionAmbiguities.estimate(
            aligned_test, reference=reference, weights=weights
        )
        standardized_test = ambiguities.standardize_product(aligned_test)

        reference_array = reference.object_.get_layers_flattened()
        test_array = standardized_test.object_.get_layers_flattened()

        common_dtype = numpy.result_type(reference_array.dtype, test_array.dtype)

        return cls(
            reference_complex=reference_array.astype(common_dtype, copy=False),
            test_complex=test_array.astype(common_dtype, copy=False),
            pixel_geometry=reference.object_.get_pixel_geometry().copy(),
            ambiguities=ambiguities,
        )

    @property
    def reference_amplitude(self) -> RealArrayType:
        """``|reference|`` — real-valued amplitude image for metrics like SSIM/PSNR."""
        return numpy.absolute(self.reference_complex)

    @property
    def test_amplitude(self) -> RealArrayType:
        """``|test|`` — real-valued amplitude image for metrics like SSIM/PSNR."""
        return numpy.absolute(self.test_complex)

    @property
    def reference_phase(self) -> RealArrayType:
        """``arg(reference)`` in radians, wrapped to ``(-pi, pi]``."""
        return numpy.angle(self.reference_complex)

    @property
    def test_phase(self) -> RealArrayType:
        """``arg(test)`` in radians, wrapped to ``(-pi, pi]``. Constant offset and
        linear ramp have been removed by :class:`ReconstructionAmbiguities`."""
        return numpy.angle(self.test_complex)


@dataclass(frozen=True)
class FourierRingCorrelation:
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

        Returns the per-ring threshold curve used by
        :meth:`get_resolution_m_at_bit_threshold`. Bins with zero pixels yield
        NaN. Useful for overlaying the threshold on an FRC plot without
        re-deriving the formula.
        """
        sigma = 0.5 * (2.0**bits - 1.0)
        sqrt_sigma = numpy.sqrt(sigma).item()
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

        if int(mask.sum()) < 2:
            return float('nan')

        f = freq[mask]
        y = corr[mask]
        span = float(f[-1] - f[0])

        if span <= 0.0:
            return float('nan')

        auc = numpy.trapezoid(y, f).item()
        return auc / span if normalize else auc

    def get_average_signal_to_noise_ratio(self) -> float:
        """Mean SSNR across bins, excluding non-finite values (NaN and +inf)."""
        ssnr = self.get_spectral_signal_to_noise_ratio()
        finite = numpy.isfinite(ssnr)
        if not bool(finite.any()):
            return float('nan')

        return numpy.mean(ssnr[finite]).item()

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

        first = int(below[0])
        # Only interpolate when the previous bin was strictly above the threshold.
        # If it merely touched the threshold (diff == 0, e.g. the DC bin where
        # FRC == 1 against the bit-threshold's N=1 limit of 1), the crossing
        # belongs at freq[first], not at the touchpoint.
        if first > 0 and finite[first - 1] and diff[first - 1] > 0.0:
            g0 = diff[first - 1].item()
            g1 = diff[first].item()
            alpha = g0 / (g0 - g1)
            crossing_freq = float(freq[first - 1]) + alpha * float(freq[first] - freq[first - 1])
        else:
            crossing_freq = freq[first].item()

        return float('nan') if crossing_freq <= 0.0 else 1.0 / crossing_freq


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
    bin_size_per_m = max(abs(kx_per_m[1].item()), abs(ky_per_m[1].item()))

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
    return numpy.sqrt(numpy.mean(numpy.square(numpy.absolute(diff)))).item()


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

    return numpy.mean(numpy.absolute(test - reference)).item()


def _infer_data_range(reference: RealArrayType) -> float:
    """Pick a sensible default ``data_range`` for PSNR/SSIM from the reference image.

    Uses ``reference.max() - reference.min()``, matching scikit-image's
    recommendation for floating-point inputs.
    """
    return numpy.ptp(reference).item()


def compute_peak_signal_to_noise_ratio(
    reference: RealArrayType,
    test: RealArrayType,
    *,
    data_range: float | None = None,
) -> float:
    """Peak signal-to-noise ratio in dB via :func:`skimage.metrics.peak_signal_noise_ratio`.

    Real-valued inputs only — project complex objects to amplitude
    (:attr:`ObjectComparison.reference_amplitude`) or phase
    (:attr:`ObjectComparison.reference_phase`) before calling.

    When ``data_range`` is ``None``, infers ``reference.max() - reference.min()``
    so floating-point inputs do not trigger scikit-image's data-range warning.
    Returns ``+inf`` when the two arrays are identical.
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    effective_range = _infer_data_range(reference) if data_range is None else data_range
    return peak_signal_noise_ratio(reference, test, data_range=effective_range).item()


def compute_structural_similarity(
    reference: RealArrayType,
    test: RealArrayType,
    *,
    data_range: float | None = None,
) -> float:
    """Structural similarity index via :func:`skimage.metrics.structural_similarity`.

    Real-valued inputs only — see :func:`compute_peak_signal_to_noise_ratio`
    for the rationale and the ``data_range`` convention. Returns a scalar in
    ``[-1, 1]``; ``1.0`` for identical inputs.
    """
    if reference.shape != test.shape:
        raise ValueError(f'Arrays must have same shape; got {reference.shape} vs {test.shape}!')

    effective_range = _infer_data_range(reference) if data_range is None else data_range
    return structural_similarity(reference, test, data_range=effective_range).item()
