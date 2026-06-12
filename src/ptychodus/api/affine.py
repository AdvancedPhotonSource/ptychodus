"""RANSAC-based affine transform estimation between two corresponding sets of probe positions."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass

import numpy

from .common import RealArrayType
from .geometry import AffineTransform
from .probe_positions import ProbePositionSequence

__all__ = ['PreprocessedCoordinates', 'estimate_affine_transform_ransac']


def _estimate_mean_hodges_lehman(values: RealArrayType) -> float:
    """Robust location estimator: median of all pairwise means. O(N^2) memory."""
    mean = numpy.median((values[numpy.newaxis, :] + values[:, numpy.newaxis]) / 2)
    return float(mean)


@dataclass(frozen=True)
class PreprocessedCoordinates:
    """Centroid-subtracted, RMS-normalized (N, 2) coordinates with the parameters needed to invert
    the normalization. Column 0 is y, column 1 is x (matching the api/probe_positions convention)."""

    coordinates: RealArrayType
    centroid_x: float
    centroid_y: float
    rms_distance: float

    @classmethod
    def from_coordinates(cls, coordinates: RealArrayType) -> PreprocessedCoordinates:
        """Build a PreprocessedCoordinates from a raw (N, 2) array packed (y, x): subtract the
        robust per-axis centroid and rescale so the resulting RMS distance from the origin is 1.
        Raises ValueError if all points are coincident."""
        centroid_y = _estimate_mean_hodges_lehman(coordinates[:, 0])
        centroid_x = _estimate_mean_hodges_lehman(coordinates[:, 1])
        centered = coordinates - numpy.array((centroid_y, centroid_x))

        distance = numpy.hypot(centered[:, 0], centered[:, 1])
        rms_distance = numpy.sqrt(numpy.mean(numpy.square(distance))).item()

        if rms_distance == 0.0:
            raise ValueError('All probe positions are coincident; cannot normalize.')

        return cls(centered / rms_distance, centroid_x, centroid_y, rms_distance)


def _estimate_affine_transform(
    uncorrected_coordinates: RealArrayType,
    corrected_coordinates: RealArrayType,
) -> AffineTransform:
    """Least-squares fit of the 6-parameter 2D affine map (uncorrected -> corrected).

    Both inputs are (N, 2) with column 0 = y, column 1 = x. The design matrix is block-structured
    so each point contributes two equations (one for x', one for y') against the six unknowns of
    ``AffineTransform`` (which always operates on physical (x, y), regardless of array packing).
    """
    n = uncorrected_coordinates.shape[0]
    y_in = uncorrected_coordinates[:, 0]
    x_in = uncorrected_coordinates[:, 1]

    a = numpy.zeros((2 * n, 6))
    a[0::2, 0] = x_in
    a[0::2, 1] = y_in
    a[0::2, 2] = 1.0
    a[1::2, 3] = x_in
    a[1::2, 4] = y_in
    a[1::2, 5] = 1.0

    b = numpy.empty(2 * n)
    b[0::2] = corrected_coordinates[:, 1]  # x'
    b[1::2] = corrected_coordinates[:, 0]  # y'

    params, *_ = numpy.linalg.lstsq(a, b, rcond=None)
    return AffineTransform(
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
        float(params[4]),
        float(params[5]),
    )


def _evaluate_error(
    uncorrected_coordinates: RealArrayType,
    corrected_coordinates: RealArrayType,
    model: AffineTransform,
) -> RealArrayType:
    """Per-point Euclidean residuals between ``model(uncorrected)`` and ``corrected``. The
    ``AffineTransform.apply_transform`` API uses (x, y) packing, so flip columns on entry/exit
    to match this module's internal (y, x) packing."""
    predicted = model.apply_transform(uncorrected_coordinates[:, ::-1])[:, ::-1]
    delta = predicted - corrected_coordinates
    return numpy.hypot(delta[:, 0], delta[:, 1])


def _is_degenerate_sample(points: RealArrayType, eps: float = 1e-10) -> bool:
    """Return True if three points are (near-)collinear and cannot pin down a full affine. The
    cross-product magnitude is axis-swap invariant, so this works for (y, x) packing too."""
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return bool(abs(cross) < eps)


def _unscale_transform(
    t_norm: AffineTransform,
    measured: PreprocessedCoordinates,
    corrected: PreprocessedCoordinates,
) -> AffineTransform:
    """Convert a transform fitted in normalized space back to the original (measured -> corrected)
    coordinate frame."""
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
    """Concatenate position sequences into a single (N, 2) array packed (y, x)."""
    coordinates: list[float] = []
    for seq in sequences:
        for point in seq:
            coordinates.append(point.coordinate_y_m)
            coordinates.append(point.coordinate_x_m)
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

    arity = 3  # minimum points needed to pin down a 2D affine (6 DOF / 2 eqns per point)
    n_points = measured.shape[0]
    if n_points < arity:
        raise ValueError(f'Need at least {arity} points to estimate an affine; got {n_points}.')

    measured_pre = PreprocessedCoordinates.from_coordinates(measured)
    corrected_pre = PreprocessedCoordinates.from_coordinates(corrected)

    best_score = numpy.inf
    best_model_norm: AffineTransform | None = None

    for _ in range(num_iterations):
        sample = rng.choice(n_points, size=arity, replace=False)
        measured_sample = measured_pre.coordinates[sample]
        corrected_sample = corrected_pre.coordinates[sample]

        if _is_degenerate_sample(measured_sample):
            continue

        coarse_model = _estimate_affine_transform(measured_sample, corrected_sample)
        all_errors = _evaluate_error(
            measured_pre.coordinates, corrected_pre.coordinates, coarse_model
        )
        inliers = numpy.flatnonzero(all_errors < inlier_threshold)

        if inliers.size < min_inliers:
            continue

        measured_inliers = measured_pre.coordinates[inliers]
        corrected_inliers = corrected_pre.coordinates[inliers]
        candidate_model = _estimate_affine_transform(measured_inliers, corrected_inliers)
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
