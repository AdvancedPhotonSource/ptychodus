"""Intensity moments of images: totals, centroids and central second moments, and the ring
assignment radial profiles bin by."""

from __future__ import annotations
from dataclasses import dataclass
import threading

import numpy

from .diffraction import DiffractionPattern, DiffractionPatterns
from .typing import BooleanArrayType, IntegerArrayType, NumberArrayType, RealArrayType

__all__ = [
    'PatternStatistics',
    'IntensityMoments',
    'compute_intensity_moments',
    'compute_ring_indexes',
]


@dataclass(frozen=True)
class IntensityMoments:
    """The zeroth, first and central second moments of intensity over the last two axes.

    Coordinates are pixels measured from the FFT origin ``N // 2`` along each axis -- a
    probe's beam axis and a detector frame's DC pixel -- x along columns and y along
    rows, so a distribution symmetric about that pixel has a zero centroid. Every field has the shape of the leading axes the
    moments were taken over, which is ``()`` for a single image. Where the total is not
    positive, the centroid and covariance are NaN, since neither is defined there.
    """

    total: RealArrayType
    centroid_x_px: RealArrayType
    centroid_y_px: RealArrayType
    covariance_xx_px2: RealArrayType
    covariance_xy_px2: RealArrayType
    covariance_yy_px2: RealArrayType


def _nan_where_empty(moment: RealArrayType, positive: BooleanArrayType) -> RealArrayType:
    """*moment* where its image holds intensity, NaN where it holds none."""
    return numpy.where(positive, moment, numpy.nan)


def compute_intensity_moments(
    intensity: NumberArrayType, *, bad_pixels: BooleanArrayType | None = None
) -> IntensityMoments:
    """Take the intensity moments of an image, or of every image in a stack.

    *intensity* has shape ``(..., height, width)`` and is used as given: no filtering,
    thresholding or background subtraction, which are the caller's to choose. Pixels
    flagged in *bad_pixels*, shape ``(height, width)``, carry no weight.

    Each moment is a sum about the array midpoint rather than about pixel zero, so a
    centred distribution does not rely on cancellation between large terms, and each
    second moment sums squared deviations from the centroid rather than subtracting its
    square from a raw moment. The sums run over the row and column marginals, with one
    batched matrix-vector product for the cross term, so no image-sized temporary is
    formed beyond the float64 weights.
    """
    height, width = intensity.shape[-2:]
    weights = numpy.asarray(intensity, dtype=numpy.float64)

    if bad_pixels is not None:
        weights = numpy.where(bad_pixels, 0.0, weights)

    x_px = numpy.arange(width, dtype=numpy.float64) - width // 2
    y_px = numpy.arange(height, dtype=numpy.float64) - height // 2

    row_sums = numpy.sum(weights, axis=-1)
    column_sums = numpy.sum(weights, axis=-2)
    total = numpy.sum(row_sums, axis=-1)
    positive = total > 0.0
    safe_total = numpy.where(positive, total, 1.0)

    centroid_x_px = (column_sums @ x_px) / safe_total
    centroid_y_px = (row_sums @ y_px) / safe_total
    deviation_x_px = x_px - centroid_x_px[..., numpy.newaxis]
    deviation_y_px = y_px - centroid_y_px[..., numpy.newaxis]

    covariance_xx_px2 = numpy.sum(column_sums * numpy.square(deviation_x_px), axis=-1)
    covariance_yy_px2 = numpy.sum(row_sums * numpy.square(deviation_y_px), axis=-1)
    row_moments_x = numpy.matmul(weights, deviation_x_px[..., numpy.newaxis])[..., 0]
    covariance_xy_px2 = numpy.sum(row_moments_x * deviation_y_px, axis=-1)

    return IntensityMoments(
        total=total,
        centroid_x_px=_nan_where_empty(centroid_x_px, positive),
        centroid_y_px=_nan_where_empty(centroid_y_px, positive),
        covariance_xx_px2=_nan_where_empty(covariance_xx_px2 / safe_total, positive),
        covariance_xy_px2=_nan_where_empty(covariance_xy_px2 / safe_total, positive),
        covariance_yy_px2=_nan_where_empty(covariance_yy_px2 / safe_total, positive),
    )


def compute_ring_indexes(
    radii: RealArrayType, bin_size: float, *, ring_edge_tolerance: float = 1.0e-9
) -> IntegerArrayType:
    """The ring each non-negative radius falls in, rings being *bin_size* wide from zero.

    Ring ``k`` holds radii in ``[k * bin_size, (k + 1) * bin_size)``. A radius that is
    an exact multiple of the bin width -- an axis pixel at a whole-pixel distance, say --
    is exact only in real arithmetic, and *ring_edge_tolerance*, in units of the bin
    width, keeps it in the ring it belongs to rather than the one below.
    """
    return numpy.floor(radii / bin_size + ring_edge_tolerance).astype(numpy.intp, copy=False)


class PatternStatistics:
    """Thread-safe running minimum, mean and maximum patterns over chunks of patterns.

    The mean is kept as a running mean rather than a running sum: each chunk is reduced
    to its own float64 mean and folded in as ``mean += (chunk_mean - mean) * n_c / (n +
    n_c)``. The stored value stays at the scale of one pattern however many are added,
    so a long stream neither overflows nor loses precision to a growing total. The
    extrema keep the pattern dtype.

    Each chunk is reduced outside the lock; the lock is held only long enough to fold
    that chunk's statistics into the running ones.
    """

    def __init__(self, height: int, width: int) -> None:
        self._lock = threading.Lock()
        self._shape = (height, width)
        self._mean_pattern = numpy.zeros((height, width), dtype=numpy.float64)
        self._minimum_pattern: DiffractionPattern | None = None
        self._maximum_pattern: DiffractionPattern | None = None
        self._num_patterns = 0

    @classmethod
    def from_patterns(
        cls,
        num_patterns: int,
        minimum_pattern: DiffractionPattern,
        mean_pattern: RealArrayType,
        maximum_pattern: DiffractionPattern,
    ) -> PatternStatistics:
        """Statistics standing for *num_patterns* patterns already reduced to these three."""
        height, width = mean_pattern.shape
        statistics = cls(height, width)

        if num_patterns > 0:
            statistics._fold(num_patterns, mean_pattern, minimum_pattern, maximum_pattern)

        return statistics

    @property
    def num_patterns(self) -> int:
        return self._num_patterns

    def add(self, patterns: DiffractionPatterns) -> None:
        """Fold a chunk of patterns into the running statistics; an empty chunk is a no-op."""
        num_patterns = patterns.shape[0]

        if num_patterns == 0:
            return

        if patterns.shape[-2:] != self._shape:
            raise ValueError(
                f'Patterns of shape {patterns.shape[-2:]} do not match the {self._shape} patterns!'
            )

        self._fold(
            num_patterns,
            numpy.mean(patterns, axis=0, dtype=numpy.float64),
            patterns.min(axis=0),
            patterns.max(axis=0),
        )

    def merge(self, other: PatternStatistics) -> None:
        """Fold another set of running statistics into these, as if its chunks were added."""
        statistics = other.get_statistics()

        if statistics is not None:
            minimum_pattern, mean_pattern, maximum_pattern = statistics
            self._fold(other.num_patterns, mean_pattern, minimum_pattern, maximum_pattern)

    def _fold(
        self,
        num_patterns: int,
        mean_pattern: RealArrayType,
        minimum_pattern: DiffractionPattern,
        maximum_pattern: DiffractionPattern,
    ) -> None:
        with self._lock:
            total = self._num_patterns + num_patterns
            self._mean_pattern += (mean_pattern - self._mean_pattern) * (num_patterns / total)
            self._num_patterns = total

            if self._minimum_pattern is None or self._maximum_pattern is None:
                self._minimum_pattern = minimum_pattern
                self._maximum_pattern = maximum_pattern
            else:
                self._minimum_pattern = numpy.minimum(self._minimum_pattern, minimum_pattern)
                self._maximum_pattern = numpy.maximum(self._maximum_pattern, maximum_pattern)

    def get_statistics(
        self,
    ) -> tuple[DiffractionPattern, RealArrayType, DiffractionPattern] | None:
        """Return copies of the ``(minimum, mean, maximum)`` patterns, or None if nothing was added.

        The mean averages the patterns actually folded in, not however many some caller
        expected, so missing chunks do not bias it toward zero.
        """
        with self._lock:
            if self._minimum_pattern is None or self._maximum_pattern is None:
                return None

            return (
                self._minimum_pattern.copy(),
                self._mean_pattern.copy(),
                self._maximum_pattern.copy(),
            )
