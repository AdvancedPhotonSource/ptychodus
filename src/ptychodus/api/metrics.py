"""Resolution metrics for reconstructed images (Fourier ring correlation, etc.)."""

from __future__ import annotations
from dataclasses import dataclass

import numpy
import scipy.fft

from .common import ComplexArrayType, IntegerArrayType, RealArrayType


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
        sigma = 0.5 * (2.0**bits - 1.0)
        sqrt_sigma = float(numpy.sqrt(sigma))
        n_per_ring = numpy.asarray(self.pixels_per_ring, dtype=float)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            inv_sqrt_n = numpy.where(n_per_ring > 0.0, 1.0 / numpy.sqrt(n_per_ring), numpy.nan)
            threshold_curve = (sigma + (2.0 * sqrt_sigma + 1.0) * inv_sqrt_n) / (
                sigma + 1.0 + 2.0 * sqrt_sigma * inv_sqrt_n
            )

        return self._resolution_at_threshold_curve(threshold_curve)

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
            g0 = float(diff[first - 1])
            g1 = float(diff[first])
            alpha = g0 / (g0 - g1)
            crossing_freq = float(freq[first - 1]) + alpha * float(freq[first] - freq[first - 1])
        else:
            crossing_freq = float(freq[first])

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
    bin_size_per_m = max(abs(float(kx_per_m[1])), abs(float(ky_per_m[1])))

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
