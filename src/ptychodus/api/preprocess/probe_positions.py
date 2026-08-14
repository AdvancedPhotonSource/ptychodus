"""Probe-position preprocessing: the affine transform primitive and RANSAC-based estimation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import overload

import numpy

from ..probe_positions import ProbePosition, ProbePositionSequence
from ..typing import RealArrayType

__all__ = [
    'AffineTransform',
    'estimate_affine_transform_ransac',
    'transform_probe_positions',
]

_AFFINE_MINIMUM_POINTS = 3  # 6 DOF / 2 equations per point


@dataclass(frozen=True)
class AffineTransform:
    """2D affine transformation expressed as a 2x3 matrix. Call it on a ``ProbePosition``
    (returns a new ``ProbePosition`` with the same index) or on an (N, 2) array packed (x, y)."""

    a00: float
    a01: float
    a02: float

    a10: float
    a11: float
    a12: float

    @overload
    def __call__(self, arg: ProbePosition) -> ProbePosition: ...
    @overload
    def __call__(self, arg: RealArrayType) -> RealArrayType: ...
    def __call__(self, arg: ProbePosition | RealArrayType) -> ProbePosition | RealArrayType:
        if isinstance(arg, ProbePosition):
            return ProbePosition(
                index=arg.index,
                coordinate_x_m=self.a00 * arg.coordinate_x_m
                + self.a01 * arg.coordinate_y_m
                + self.a02,
                coordinate_y_m=self.a10 * arg.coordinate_x_m
                + self.a11 * arg.coordinate_y_m
                + self.a12,
            )
        linear = numpy.array([[self.a00, self.a01], [self.a10, self.a11]])
        translation = numpy.array([self.a02, self.a12])
        return arg @ linear.T + translation


def _estimate_mean_hodges_lehmann(values: RealArrayType) -> float:
    """Robust location estimator: median of all pairwise means. O(N^2) memory."""
    return float(numpy.median(numpy.add.outer(values, values) / 2))


@dataclass(frozen=True)
class _PreprocessedCoordinates:
    """Centroid-subtracted, RMS-normalized (N, 2) coordinates with the parameters needed to invert
    the normalization. Column 0 is x, column 1 is y."""

    coordinates: RealArrayType
    centroid_x: float
    centroid_y: float
    rms_distance: float


def _preprocess_coordinates(coordinates: RealArrayType) -> _PreprocessedCoordinates:
    """Centroid-subtract and RMS-normalize a raw (N, 2) array packed (x, y): subtract the robust
    per-axis centroid and rescale so the resulting RMS distance from the origin is 1. Raises
    ValueError if the shape is wrong or all points are coincident."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f'Expected (N, 2) coordinate array; got shape {coordinates.shape}.')

    centroid_x = _estimate_mean_hodges_lehmann(coordinates[:, 0])
    centroid_y = _estimate_mean_hodges_lehmann(coordinates[:, 1])
    centered = coordinates - numpy.array((centroid_x, centroid_y))

    distance = numpy.hypot(centered[:, 0], centered[:, 1])
    rms_distance = numpy.sqrt(numpy.mean(numpy.square(distance))).item()

    if rms_distance == 0.0:
        raise ValueError('All probe positions are coincident; cannot normalize.')

    return _PreprocessedCoordinates(centered / rms_distance, centroid_x, centroid_y, rms_distance)


def transform_probe_positions(
    positions: Iterable[ProbePosition],
    transform: AffineTransform,
    rng: numpy.random.Generator | None = None,
    jitter_radius_m: float = 0.0,
) -> Iterator[ProbePosition]:
    """Apply an affine transform (and optional random jitter) to every position in *positions*."""
    for position in positions:
        transformed = transform(position)

        if rng is not None:
            angle_rad = 2 * numpy.pi * rng.uniform()
            radius_m = jitter_radius_m * numpy.sqrt(rng.uniform())
            transformed = ProbePosition(
                index=transformed.index,
                coordinate_x_m=transformed.coordinate_x_m + radius_m * numpy.cos(angle_rad),
                coordinate_y_m=transformed.coordinate_y_m + radius_m * numpy.sin(angle_rad),
            )

        yield transformed


def _fit_affine_least_squares(
    uncorrected_coordinates: RealArrayType,
    corrected_coordinates: RealArrayType,
) -> AffineTransform:
    """Least-squares fit of the 6-parameter 2D affine map (uncorrected -> corrected).

    Both inputs are (N, 2) with column 0 = x, column 1 = y. Each point contributes a row
    ``[x, y, 1]`` to the design matrix; the x' and y' fits share this design and are solved
    together via a multi-column RHS.
    """
    n = uncorrected_coordinates.shape[0]
    design = numpy.column_stack((uncorrected_coordinates, numpy.ones(n)))  # (N, 3): [x, y, 1]
    params, *_ = numpy.linalg.lstsq(design, corrected_coordinates, rcond=None)  # (3, 2)
    (a00, a10), (a01, a11), (a02, a12) = params
    return AffineTransform(a00, a01, a02, a10, a11, a12)


def _evaluate_error(
    uncorrected_coordinates: RealArrayType,
    corrected_coordinates: RealArrayType,
    model: AffineTransform,
) -> RealArrayType:
    """Per-point Euclidean residuals between ``model(uncorrected)`` and ``corrected``."""
    delta = model(uncorrected_coordinates) - corrected_coordinates
    return numpy.hypot(delta[:, 0], delta[:, 1])


def _is_degenerate_sample(points: RealArrayType, eps: float = 1e-10) -> bool:
    """Return True if three points are (near-)collinear and cannot pin down a full affine."""
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return bool(abs(cross) < eps)


def _unscale_transform(
    t_norm: AffineTransform,
    measured: _PreprocessedCoordinates,
    corrected: _PreprocessedCoordinates,
) -> AffineTransform:
    """Convert a transform fitted in normalized (centroid-subtracted, RMS-scaled) space back to
    the original (measured -> corrected) (x, y) coordinate frame."""
    s = corrected.rms_distance / measured.rms_distance
    a = s * t_norm.a00
    b = s * t_norm.a01
    d = s * t_norm.a10
    e = s * t_norm.a11
    tx = (
        corrected.rms_distance * t_norm.a02
        + corrected.centroid_x
        - a * measured.centroid_x
        - b * measured.centroid_y
    )
    ty = (
        corrected.rms_distance * t_norm.a12
        + corrected.centroid_y
        - d * measured.centroid_x
        - e * measured.centroid_y
    )
    return AffineTransform(a, b, tx, d, e, ty)


def _flatten_to_array(sequences: Iterable[ProbePositionSequence]) -> RealArrayType:
    """Concatenate position sequences into a single (N, 2) array packed (x, y)."""
    coordinates: list[float] = []
    for seq in sequences:
        for point in seq:
            coordinates.append(point.coordinate_x_m)
            coordinates.append(point.coordinate_y_m)
    return numpy.reshape(coordinates, (-1, 2))


def estimate_affine_transform_ransac(
    uncorrected_positions: Iterable[ProbePositionSequence],
    corrected_positions: Iterable[ProbePositionSequence],
    *,
    num_iterations: int = 1000,
    inlier_threshold: float = 0.05,
    min_inliers: int = 10,
    rng: numpy.random.Generator | None = None,
) -> AffineTransform:
    """Estimate the affine transform that best maps ``uncorrected_positions`` onto
    ``corrected_positions`` using RANSAC with a robust (Hodges-Lehmann centroid + unit-RMS)
    normalization.

    Each iterable contributes one or more ``ProbePositionSequence`` objects; the points are
    concatenated in iteration order and paired one-to-one between the two sides (so the totals
    must match).

    Args:
        uncorrected_positions: Source point sequences (typically the raw scan positions).
        corrected_positions: Target point sequences (typically the ptychography-refined positions).
        num_iterations: Number of RANSAC minimal samples to draw. Defaults to 1000.
        inlier_threshold: Maximum residual (in unit-RMS normalized space) for a point to count as
            an inlier. Defaults to 0.05 (~5% of the typical scan extent).
        min_inliers: Minimum inlier count required before a candidate model is scored.
            Defaults to 10.
        rng: NumPy random generator. If None, a fresh ``numpy.random.default_rng()`` is created.

    Returns:
        The best-scoring affine transform in the original (un-normalized) coordinate frame.

    Raises:
        ValueError: If the two sides have different total point counts, or if fewer than 3 points
            are supplied (an affine transform has 6 DOF and needs at least 3 non-collinear points).
        RuntimeError: If no candidate model reaches ``min_inliers`` inliers within
            ``inlier_threshold`` over ``num_iterations`` iterations.
    """
    if rng is None:
        rng = numpy.random.default_rng()

    measured = _flatten_to_array(uncorrected_positions)
    corrected = _flatten_to_array(corrected_positions)

    if measured.shape[0] != corrected.shape[0]:
        raise ValueError(
            'Measured and corrected coordinate sets have different lengths '
            f'({measured.shape[0]} vs {corrected.shape[0]}); '
            'point-by-point correspondence is required.'
        )

    n_points = measured.shape[0]
    if n_points < _AFFINE_MINIMUM_POINTS:
        raise ValueError(
            f'Need at least {_AFFINE_MINIMUM_POINTS} points to estimate an affine; got {n_points}.'
        )

    measured_pre = _preprocess_coordinates(measured)
    corrected_pre = _preprocess_coordinates(corrected)

    best_score = numpy.inf
    best_model_norm: AffineTransform | None = None

    for _ in range(num_iterations):
        sample = rng.choice(n_points, size=_AFFINE_MINIMUM_POINTS, replace=False)
        measured_sample = measured_pre.coordinates[sample]
        corrected_sample = corrected_pre.coordinates[sample]

        if _is_degenerate_sample(measured_sample):
            continue

        coarse_model = _fit_affine_least_squares(measured_sample, corrected_sample)
        all_errors = _evaluate_error(
            measured_pre.coordinates, corrected_pre.coordinates, coarse_model
        )
        inliers = numpy.flatnonzero(all_errors < inlier_threshold)

        if inliers.size < min_inliers:
            continue

        measured_inliers = measured_pre.coordinates[inliers]
        corrected_inliers = corrected_pre.coordinates[inliers]
        candidate_model = _fit_affine_least_squares(measured_inliers, corrected_inliers)
        candidate_errors = _evaluate_error(measured_inliers, corrected_inliers, candidate_model)
        candidate_rms = float(numpy.sqrt(numpy.mean(numpy.square(candidate_errors))))

        if candidate_rms < best_score:
            best_score = candidate_rms
            best_model_norm = candidate_model

    if best_model_norm is None:
        raise RuntimeError(
            f'RANSAC did not find a model with at least {min_inliers} inliers within '
            f'threshold {inlier_threshold} in {num_iterations} iterations.'
        )

    return _unscale_transform(best_model_norm, measured_pre, corrected_pre)
