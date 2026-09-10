"""Robust noise-floor estimation for background subtraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy

from ..geometry import Interval
from ..typing import RealArrayType


@dataclass(frozen=True)
class RobustStatistics:
    """Robust central-tendency and dispersion estimate: median and the median
    of the absolute deviations from that median (MAD)."""

    median: float
    median_absolute_deviation: float

    def get_significance_threshold(self, k: float) -> float:
        """values at or above the significance threshold are considered statistically distinct
        from the noise floor. ``k`` is a unitless multiplier; larger values are more
        aggressive at suppressing noise tails but increasingly truncate real signal in the wings.
        """
        return self.median + k * self.median_absolute_deviation

    def get_bounds(self, k: float, *, require_positive: bool = False) -> Interval[float]:
        """Symmetric closed interval [median - k*MAD, median + k*MAD].

        Two-sided companion to :meth:`get_significance_threshold`. ``k`` is a
        unitless multiplier; larger ``k`` widens the interval.

        When ``require_positive`` is True the lower bound is clipped up to
        ``numpy.nextafter(0.0, 1.0)`` so the interval excludes zero and
        negative values. This constraint applies independently of the
        dispersion, so it also fires in the zero-MAD degenerate case.

        Zero-MAD degenerate cases return an interval that is unbounded on the
        sides where no bound can be derived:

        - ``MAD == 0`` with ``require_positive=False`` -> ``Interval(-inf, +inf)``.
        - ``MAD == 0`` with ``require_positive=True``  -> ``Interval(nextafter(0, 1), +inf)``.
        """
        positive_lower = float(numpy.nextafter(0.0, 1.0))
        if self.median_absolute_deviation == 0.0:
            lower = positive_lower if require_positive else -numpy.inf
            return Interval[float](lower, numpy.inf)
        lower = self.median - k * self.median_absolute_deviation
        upper = self.median + k * self.median_absolute_deviation
        if require_positive:
            lower = max(lower, positive_lower)
        return Interval[float](lower, upper)


def compute_robust_statistics(values: RealArrayType) -> RobustStatistics:
    """Return the median and MAD of ``values``. Caller must ensure ``values`` is non-empty."""
    median = numpy.median(values)
    absolute_deviation = numpy.abs(values - median)
    return RobustStatistics(
        median=float(median),
        median_absolute_deviation=float(numpy.median(absolute_deviation)),
    )


def estimate_noise_floor(
    values: RealArrayType,
    *,
    fallback_values: RealArrayType | None = None,
    num_bins: int = 256,
    bimodality_threshold: float = 0.75,
) -> RobustStatistics:
    """Estimate the background level and noise scale of a pool of intensities.

    Strategy:

    1. Compute Otsu's threshold on the intensity histogram of *values* and
       Otsu's class-separability measure ``eta = sigma_B^2 / sigma_T^2`` (the
       between-class variance at the optimal threshold divided by the total
       variance, both computed on the same histogram).
    2. If ``eta >= bimodality_threshold`` the histogram is sufficiently
       bimodal: take the background pool to be the values below Otsu's
       threshold and return the median and MAD of that pool.
    3. Otherwise the histogram is unimodal — Otsu has no meaningful signal /
       background split — so fall back to the median and MAD of
       *fallback_values* (or of *values* if no fallback is supplied).

    Pixel ordering and shape are ignored — only the values matter.

    The default ``bimodality_threshold`` of 0.75 is set comfortably above the
    Otsu separability of a pure Gaussian (``eta = 2 / pi ~ 0.637``) so that a
    noise-only histogram correctly triggers the fallback. Real signal +
    background mixtures typically have ``eta >= 0.85``.
    """
    flat = values.ravel()
    fallback = fallback_values.ravel() if fallback_values is not None else flat

    if flat.size == 0:
        return compute_robust_statistics(fallback)

    pmin = flat.min()
    pmax = flat.max()

    if pmax <= pmin:
        # Degenerate histogram (single intensity value): Otsu has no split.
        return compute_robust_statistics(fallback)

    hist, edges = numpy.histogram(flat, bins=num_bins, range=(pmin, pmax))
    total_count = hist.sum()

    if total_count == 0:
        return compute_robust_statistics(fallback)

    # Otsu: maximize between-class variance.
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    probability = hist.astype(numpy.float64) / total_count
    cumulative_weight = numpy.cumsum(probability)
    cumulative_mean = numpy.cumsum(probability * bin_centers)
    total_mean = cumulative_mean[-1]

    # sigma_B^2(t) = (mu_T * w(t) - mu(t))^2 / (w(t) * (1 - w(t)))
    denominator = cumulative_weight * (1.0 - cumulative_weight)
    numerator = (total_mean * cumulative_weight - cumulative_mean) ** 2
    between_class_variance = numpy.divide(
        numerator,
        denominator,
        out=numpy.zeros_like(numerator),
        where=denominator > 0.0,
    )

    total_variance = numpy.sum(probability * (bin_centers - total_mean) ** 2)

    best_bin = numpy.argmax(between_class_variance)
    best_between_class_variance = between_class_variance[best_bin]
    separability = best_between_class_variance / total_variance if total_variance > 0.0 else 0.0

    if separability < bimodality_threshold:
        return compute_robust_statistics(fallback)

    otsu_threshold = edges[best_bin + 1]
    background_pool = flat[flat < otsu_threshold]

    if background_pool.size == 0:
        return compute_robust_statistics(fallback)

    return compute_robust_statistics(background_pool)
