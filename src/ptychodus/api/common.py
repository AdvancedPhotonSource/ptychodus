"""Common type aliases, physical constants, and utility functions used throughout the API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, overload

import numpy
import numpy.typing

# Type Aliases
IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]
InexactArrayType: TypeAlias = numpy.typing.NDArray[numpy.inexact[Any]]
NumberArrayType: TypeAlias = numpy.typing.NDArray[numpy.number]

# Mathematical Constants
BYTES_PER_MEGABYTE: Final[int] = 1000 * 1000
TWO_PI: Final[float] = 2.0 * numpy.pi
TWO_PI_J: Final[complex] = 2.0j * numpy.pi

# Physical Constants
# Source: https://physics.nist.gov/cuu/Constants/index.html
ELECTRON_VOLT_J: Final[float] = 1.602176634e-19
LIGHT_SPEED_M_PER_S: Final[float] = 299792458
PLANCK_CONSTANT_J_PER_HZ: Final[float] = 6.62607015e-34


def get_ptychodus_dir() -> Path:
    """Return the user's Ptychodus configuration directory (``~/.ptychodus``)."""
    return Path.home() / '.ptychodus'


@overload
def lerp(lower: float, upper: float, frac: float) -> float:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: complex, upper: complex, frac: float) -> complex:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: float) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: RealArrayType) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: float, upper: float, frac: RealArrayType) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


def lerp(
    lower: InexactArrayType | complex,
    upper: InexactArrayType | complex,
    frac: RealArrayType | float,
) -> InexactArrayType | complex:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    return (1.0 - frac) * lower + frac * upper


@dataclass(frozen=True)
class NoiseFloor:
    """Robust noise-floor estimate: background value and its median absolute deviation."""

    background_value: float
    median_absolute_deviation: float

    @classmethod
    def from_values(cls, values: RealArrayType) -> NoiseFloor:
        background_value = numpy.median(values)
        absolute_deviation = numpy.abs(values - background_value)
        return cls(
            background_value=background_value.item(),
            median_absolute_deviation=numpy.median(absolute_deviation).item(),
        )

    def get_significance_threshold(self, mad_threshold: float) -> float:
        """values at or above the significance threshold are considered statistically distinct
        from the noise floor. ``mad_threshold`` is a unitless multiplier; larger values are more
        aggressive at suppressing noise tails but increasingly truncate real signal in the wings.
        """
        return self.background_value + mad_threshold * self.median_absolute_deviation


def estimate_noise_floor(
    values: RealArrayType,
    *,
    fallback_values: RealArrayType | None = None,
    num_bins: int = 256,
    bimodality_threshold: float = 0.75,
) -> NoiseFloor:
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
        return NoiseFloor.from_values(fallback)

    pmin = flat.min().item()
    pmax = flat.max().item()

    if pmax <= pmin:
        # Degenerate histogram (single intensity value): Otsu has no split.
        return NoiseFloor.from_values(fallback)

    hist, edges = numpy.histogram(flat, bins=num_bins, range=(pmin, pmax))
    total_count = int(hist.sum())

    if total_count == 0:
        return NoiseFloor.from_values(fallback)

    # Otsu: maximize between-class variance.
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    probability = hist.astype(numpy.float64) / total_count
    cumulative_weight = numpy.cumsum(probability)
    cumulative_mean = numpy.cumsum(probability * bin_centers)
    total_mean = float(cumulative_mean[-1])

    # sigma_B^2(t) = (mu_T * w(t) - mu(t))^2 / (w(t) * (1 - w(t)))
    denominator = cumulative_weight * (1.0 - cumulative_weight)
    numerator = (total_mean * cumulative_weight - cumulative_mean) ** 2
    between_class_variance = numpy.divide(
        numerator,
        denominator,
        out=numpy.zeros_like(numerator),
        where=denominator > 0.0,
    )

    total_variance = float(numpy.sum(probability * (bin_centers - total_mean) ** 2))

    best_bin = int(numpy.argmax(between_class_variance))
    best_between_class_variance = float(between_class_variance[best_bin])
    separability = best_between_class_variance / total_variance if total_variance > 0.0 else 0.0

    if separability < bimodality_threshold:
        return NoiseFloor.from_values(fallback)

    otsu_threshold = float(edges[best_bin + 1])
    background_pool = flat[flat < otsu_threshold]

    if background_pool.size == 0:
        return NoiseFloor.from_values(fallback)

    return NoiseFloor.from_values(background_pool)
