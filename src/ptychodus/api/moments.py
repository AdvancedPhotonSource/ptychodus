"""Intensity moments of images: totals, centroids and central second moments."""

from __future__ import annotations
from dataclasses import dataclass
import threading

import numpy

from .diffraction import DiffractionPattern, DiffractionPatterns
from .typing import BooleanArrayType, NumberArrayType, RealArrayType

__all__ = [
    'FrameStatistics',
    'IntensityMoments',
    'compute_intensity_moments',
]


@dataclass(frozen=True)
class IntensityMoments:
    """The zeroth, first and central second moments of intensity over the last two axes.

    Coordinates are pixels measured from the array midpoint ``(N - 1) / 2`` along each
    axis, x along columns and y along rows, so a distribution symmetric about the middle
    of the array has a zero centroid. Every field has the shape of the leading axes the
    moments were taken over, which is ``()`` for a single image. Where the total is not
    positive, the centroid and covariance are NaN, since neither is defined there.
    """

    total: RealArrayType
    centroid_x_px: RealArrayType
    centroid_y_px: RealArrayType
    covariance_xx_px2: RealArrayType
    covariance_xy_px2: RealArrayType
    covariance_yy_px2: RealArrayType


def compute_intensity_moments(
    intensity: NumberArrayType, *, bad_pixels: BooleanArrayType | None = None
) -> IntensityMoments:
    """Take the intensity moments of an image, or of every image in a stack.

    *intensity* has shape ``(..., height, width)`` and is used as given: no filtering,
    thresholding or background subtraction, which are the caller's to choose. Pixels
    flagged in *bad_pixels*, shape ``(height, width)``, carry no weight.

    Each moment is a sum about the array midpoint rather than about pixel zero, so a
    centred distribution does not rely on cancellation between large terms.
    """
    height, width = intensity.shape[-2:]
    weights = intensity.astype(numpy.float64)

    if bad_pixels is not None:
        weights = numpy.where(bad_pixels, 0.0, weights)

    x_px = numpy.arange(width, dtype=numpy.float64) - (width - 1) / 2
    y_px = (numpy.arange(height, dtype=numpy.float64) - (height - 1) / 2).reshape(-1, 1)

    total = numpy.sum(weights, axis=(-2, -1))
    positive = total > 0.0
    safe_total = numpy.where(positive, total, 1.0)

    centroid_x_px = numpy.sum(weights * x_px, axis=(-2, -1)) / safe_total
    centroid_y_px = numpy.sum(weights * y_px, axis=(-2, -1)) / safe_total
    deviation_x_px = x_px - centroid_x_px[..., numpy.newaxis, numpy.newaxis]
    deviation_y_px = y_px - centroid_y_px[..., numpy.newaxis, numpy.newaxis]

    covariance_xx_px2 = numpy.sum(weights * numpy.square(deviation_x_px), axis=(-2, -1))
    covariance_xy_px2 = numpy.sum(weights * deviation_x_px * deviation_y_px, axis=(-2, -1))
    covariance_yy_px2 = numpy.sum(weights * numpy.square(deviation_y_px), axis=(-2, -1))

    def _defined(moment: RealArrayType) -> RealArrayType:
        return numpy.where(positive, moment, numpy.nan)

    return IntensityMoments(
        total=total,
        centroid_x_px=_defined(centroid_x_px),
        centroid_y_px=_defined(centroid_y_px),
        covariance_xx_px2=_defined(covariance_xx_px2 / safe_total),
        covariance_xy_px2=_defined(covariance_xy_px2 / safe_total),
        covariance_yy_px2=_defined(covariance_yy_px2 / safe_total),
    )


class FrameStatistics:
    """Thread-safe running minimum, mean and maximum frames over chunks of patterns.

    The mean is kept as a running mean rather than a running sum: each chunk is reduced
    to its own float64 mean and folded in as ``mean += (chunk_mean - mean) * n_c / (n +
    n_c)``. The stored value stays at the scale of one pattern however many are added,
    so a long stream neither overflows nor loses precision to a growing total. The
    extrema keep the pattern dtype.

    Each chunk is reduced outside the lock; the lock is held only long enough to fold
    that chunk's frames into the running ones.
    """

    def __init__(self, height: int, width: int) -> None:
        self._lock = threading.Lock()
        self._shape = (height, width)
        self._mean_frame = numpy.zeros((height, width), dtype=numpy.float64)
        self._minimum_frame: DiffractionPattern | None = None
        self._maximum_frame: DiffractionPattern | None = None
        self._num_patterns = 0

    @classmethod
    def from_frames(
        cls,
        num_patterns: int,
        minimum_frame: DiffractionPattern,
        mean_frame: RealArrayType,
        maximum_frame: DiffractionPattern,
    ) -> FrameStatistics:
        """Statistics standing for *num_patterns* patterns already reduced to these frames."""
        height, width = mean_frame.shape
        statistics = cls(height, width)

        if num_patterns > 0:
            statistics._fold(num_patterns, mean_frame, minimum_frame, maximum_frame)

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
                f'Patterns of shape {patterns.shape[-2:]} do not match the {self._shape} frames!'
            )

        self._fold(
            num_patterns,
            numpy.mean(patterns, axis=0, dtype=numpy.float64),
            patterns.min(axis=0),
            patterns.max(axis=0),
        )

    def merge(self, other: FrameStatistics) -> None:
        """Fold another set of running statistics into these, as if its chunks were added."""
        frames = other.get_frames()

        if frames is not None:
            minimum_frame, mean_frame, maximum_frame = frames
            self._fold(other.num_patterns, mean_frame, minimum_frame, maximum_frame)

    def _fold(
        self,
        num_patterns: int,
        mean_frame: RealArrayType,
        minimum_frame: DiffractionPattern,
        maximum_frame: DiffractionPattern,
    ) -> None:
        with self._lock:
            total = self._num_patterns + num_patterns
            self._mean_frame += (mean_frame - self._mean_frame) * (num_patterns / total)
            self._num_patterns = total

            if self._minimum_frame is None or self._maximum_frame is None:
                self._minimum_frame = minimum_frame
                self._maximum_frame = maximum_frame
            else:
                self._minimum_frame = numpy.minimum(self._minimum_frame, minimum_frame)
                self._maximum_frame = numpy.maximum(self._maximum_frame, maximum_frame)

    def get_frames(
        self,
    ) -> tuple[DiffractionPattern, RealArrayType, DiffractionPattern] | None:
        """Return copies of the ``(minimum, mean, maximum)`` frames, or None if nothing was added.

        The mean averages the patterns actually folded in, not however many some caller
        expected, so missing chunks do not bias it toward zero.
        """
        with self._lock:
            if self._minimum_frame is None or self._maximum_frame is None:
                return None

            return (
                self._minimum_frame.copy(),
                self._mean_frame.copy(),
                self._maximum_frame.copy(),
            )
