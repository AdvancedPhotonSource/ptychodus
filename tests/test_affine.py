from typing import cast
from unittest.mock import MagicMock

import numpy
import pytest

from ptychodus.api.preprocess.probe_positions import (
    _estimate_mean_hodges_lehmann,
    _evaluate_error,
    _fit_affine_least_squares,
    _preprocess_coordinates,
    _unscale_transform,
    estimate_affine_transform_ransac,
)
from ptychodus.api.preprocess.probe_positions import AffineTransform
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.model.analysis.affine import AffineTransformEstimator
from ptychodus.model.analysis.settings import AffineTransformEstimatorSettings
from ptychodus.model.product import ProbePositionsRepository


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _params(t: AffineTransform) -> tuple[float, float, float, float, float, float]:
    return (t.a00, t.a01, t.a02, t.a10, t.a11, t.a12)


def _sequence_from_xy(arr: numpy.ndarray) -> ProbePositionSequence:
    """Build a ProbePositionSequence from an (N, 2) array packed (x, y)."""
    return ProbePositionSequence(
        [
            ProbePosition(index=i, coordinate_x_m=float(arr[i, 0]), coordinate_y_m=float(arr[i, 1]))
            for i in range(arr.shape[0])
        ]
    )


# ---------------------------------------------------------------------------
# _fit_affine_least_squares: pure least-squares fit
# ---------------------------------------------------------------------------


def test_fit_affine_recovers_known_transform() -> None:
    """Given clean point correspondences, lstsq recovers the exact 6-parameter affine."""
    truth = AffineTransform(1.7, -0.3, 0.4, 0.25, 2.1, -1.6)
    rng = numpy.random.default_rng(42)
    uncorrected = rng.uniform(-10.0, 10.0, size=(20, 2))
    corrected = truth(uncorrected)

    recovered = _fit_affine_least_squares(uncorrected, corrected)

    numpy.testing.assert_allclose(_params(recovered), _params(truth), atol=1e-10)


def test_fit_affine_three_points_is_exact() -> None:
    """With exactly 3 non-collinear points (6 equations, 6 unknowns), the fit is exact."""
    truth = AffineTransform(2.0, 0.5, 1.0, -0.5, 3.0, -2.0)
    uncorrected = numpy.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    corrected = truth(uncorrected)

    recovered = _fit_affine_least_squares(uncorrected, corrected)

    numpy.testing.assert_allclose(_params(recovered), _params(truth), atol=1e-12)


# ---------------------------------------------------------------------------
# _estimate_mean_hodges_lehmann: robust location estimator
# ---------------------------------------------------------------------------


def test_hodges_lehmann_clean_arange() -> None:
    """For a symmetric uniform sample, the H-L mean equals the median."""
    values = numpy.arange(11, dtype=float)
    assert _estimate_mean_hodges_lehmann(values) == pytest.approx(5.0)


def test_hodges_lehmann_resists_outliers() -> None:
    """A single extreme outlier shifts the arithmetic mean but barely moves the H-L mean."""
    clean = numpy.arange(11, dtype=float)
    contaminated = numpy.append(clean, 1000.0)
    arithmetic_mean = float(numpy.mean(contaminated))
    hl_mean = _estimate_mean_hodges_lehmann(contaminated)
    assert arithmetic_mean > 80.0  # the outlier dominates
    assert abs(hl_mean - 5.0) < 1.0  # H-L estimate stays near the true center


# ---------------------------------------------------------------------------
# _preprocess_coordinates: shape validation
# ---------------------------------------------------------------------------


def test_preprocess_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match='Expected .N, 2. coordinate array'):
        _preprocess_coordinates(numpy.zeros((5, 3)))
    with pytest.raises(ValueError, match='Expected .N, 2. coordinate array'):
        _preprocess_coordinates(numpy.zeros(6))


def test_preprocess_rejects_coincident_points() -> None:
    with pytest.raises(ValueError, match='coincident'):
        _preprocess_coordinates(numpy.zeros((5, 2)))


# ---------------------------------------------------------------------------
# _unscale_transform: round-trip from normalized to original coordinates
# ---------------------------------------------------------------------------


def test_unscale_identity_normalizes_to_normalization() -> None:
    """If T_norm is the identity, the unscaled transform must map measured -> corrected
    using exactly the centroid/RMS reparametrization."""
    rng = numpy.random.default_rng(0)
    raw_measured = rng.uniform(-5.0, 5.0, size=(15, 2))
    raw_corrected = rng.uniform(-5.0, 5.0, size=(15, 2))

    measured_pre = _preprocess_coordinates(raw_measured)
    corrected_pre = _preprocess_coordinates(raw_corrected)

    identity = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    t = _unscale_transform(identity, measured_pre, corrected_pre)

    # Identity in normalized space means: shift to origin, rescale to corrected RMS, shift to
    # corrected centroid. Apply that manually under (x, y) column packing and compare.
    s = corrected_pre.rms_distance / measured_pre.rms_distance
    expected = numpy.column_stack(
        (
            s * (raw_measured[:, 0] - measured_pre.centroid_x) + corrected_pre.centroid_x,
            s * (raw_measured[:, 1] - measured_pre.centroid_y) + corrected_pre.centroid_y,
        )
    )
    actual = t(raw_measured)
    numpy.testing.assert_allclose(actual, expected, atol=1e-10)


def test_unscale_recovers_transform_through_normalized_fit() -> None:
    """Fit in normalized space and unscale; the result must equal the transform fit directly."""
    truth = AffineTransform(1.4, -0.2, 3.5, 0.1, 0.9, -7.0)
    rng = numpy.random.default_rng(1)
    raw_measured = rng.uniform(-50.0, 50.0, size=(30, 2))
    raw_corrected = truth(raw_measured)

    measured_pre = _preprocess_coordinates(raw_measured)
    corrected_pre = _preprocess_coordinates(raw_corrected)

    t_norm = _fit_affine_least_squares(measured_pre.coordinates, corrected_pre.coordinates)
    recovered = _unscale_transform(t_norm, measured_pre, corrected_pre)

    numpy.testing.assert_allclose(_params(recovered), _params(truth), atol=1e-8)


# ---------------------------------------------------------------------------
# estimate_affine_transform_ransac: end-to-end api entrypoint
# ---------------------------------------------------------------------------


def test_ransac_recovers_transform_with_outliers() -> None:
    """With 20% gross outliers, RANSAC still pins the affine within tight tolerance."""
    rng = numpy.random.default_rng(2026)
    n = 100
    truth = AffineTransform(1.05, 0.02, 1e-5, -0.03, 0.98, -2e-5)

    measured = rng.uniform(-1e-4, 1e-4, size=(n, 2))
    corrected = truth(measured)

    n_outliers = 20
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    corrected[outlier_idx] += rng.uniform(-1e-3, 1e-3, size=(n_outliers, 2))

    result = estimate_affine_transform_ransac(
        [_sequence_from_xy(measured)],
        [_sequence_from_xy(corrected)],
        num_iterations=200,
        inlier_threshold=0.01,
        min_inliers=50,
        rng=numpy.random.default_rng(7),
    )

    # Linear part: very close to truth.
    numpy.testing.assert_allclose(
        [result.a00, result.a01, result.a10, result.a11],
        [truth.a00, truth.a01, truth.a10, truth.a11],
        atol=5e-3,
    )
    # Translation: scaled by the scan extent (~1e-4), so allow proportional tolerance.
    assert abs(result.a02 - truth.a02) < 1e-6
    assert abs(result.a12 - truth.a12) < 1e-6


def test_ransac_concatenates_multiple_sequences() -> None:
    """Two sequences on each side should be concatenated in iteration order."""
    rng = numpy.random.default_rng(99)
    truth = AffineTransform(1.2, 0.0, 5.0, 0.0, 1.2, -3.0)

    measured_a = rng.uniform(-1.0, 1.0, size=(15, 2))
    measured_b = rng.uniform(-1.0, 1.0, size=(15, 2))
    corrected_a = truth(measured_a)
    corrected_b = truth(measured_b)

    result = estimate_affine_transform_ransac(
        [_sequence_from_xy(measured_a), _sequence_from_xy(measured_b)],
        [_sequence_from_xy(corrected_a), _sequence_from_xy(corrected_b)],
        num_iterations=100,
        inlier_threshold=0.01,
        min_inliers=20,
        rng=numpy.random.default_rng(0),
    )

    numpy.testing.assert_allclose(_params(result), _params(truth), atol=1e-6)


def test_ransac_mismatched_lengths_raises() -> None:
    """Total point counts on each side must match."""
    rng = numpy.random.default_rng(4)
    measured = rng.uniform(-1.0, 1.0, size=(10, 2))
    corrected = rng.uniform(-1.0, 1.0, size=(7, 2))

    with pytest.raises(ValueError, match='different lengths'):
        estimate_affine_transform_ransac(
            [_sequence_from_xy(measured)],
            [_sequence_from_xy(corrected)],
            rng=numpy.random.default_rng(0),
        )


def test_ransac_too_few_points_raises() -> None:
    """Need at least 3 points to estimate an affine."""
    rng = numpy.random.default_rng(5)
    measured = rng.uniform(-1.0, 1.0, size=(2, 2))
    corrected = rng.uniform(-1.0, 1.0, size=(2, 2))

    with pytest.raises(ValueError, match='at least 3 points'):
        estimate_affine_transform_ransac(
            [_sequence_from_xy(measured)],
            [_sequence_from_xy(corrected)],
            rng=numpy.random.default_rng(0),
        )


def test_ransac_no_inliers_raises() -> None:
    """If RANSAC never finds min_inliers within threshold, surface a RuntimeError."""
    rng = numpy.random.default_rng(3)
    measured = rng.uniform(-1.0, 1.0, size=(20, 2))
    # Garbage correspondences guarantee large per-point residuals.
    corrected = rng.uniform(-1.0, 1.0, size=(20, 2))

    with pytest.raises(RuntimeError, match='RANSAC did not find'):
        estimate_affine_transform_ransac(
            [_sequence_from_xy(measured)],
            [_sequence_from_xy(corrected)],
            num_iterations=20,
            inlier_threshold=1e-10,
            min_inliers=15,
            rng=numpy.random.default_rng(0),
        )


def test_ransac_default_rng_runs() -> None:
    """Smoke test that the rng=None default path works (auto-creates a generator)."""
    truth = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # identity
    rng = numpy.random.default_rng(6)
    measured = rng.uniform(-1.0, 1.0, size=(30, 2))
    corrected = truth(measured)

    result = estimate_affine_transform_ransac(
        [_sequence_from_xy(measured)],
        [_sequence_from_xy(corrected)],
        num_iterations=50,
        inlier_threshold=0.01,
        min_inliers=10,
    )

    numpy.testing.assert_allclose(_params(result), _params(truth), atol=1e-6)


# ---------------------------------------------------------------------------
# _evaluate_error sanity checks
# ---------------------------------------------------------------------------


def test_evaluate_error_zero_for_perfect_model() -> None:
    truth = AffineTransform(1.5, 0.1, -0.7, -0.2, 1.3, 4.2)
    rng = numpy.random.default_rng(11)
    measured = rng.uniform(-3.0, 3.0, size=(8, 2))
    corrected = truth(measured)

    errors = _evaluate_error(measured, corrected, truth)

    numpy.testing.assert_allclose(errors, numpy.zeros(8), atol=1e-12)


def test_evaluate_error_matches_pointwise_distance() -> None:
    model = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # identity
    measured = numpy.array([[0.0, 0.0], [3.0, 4.0]])
    corrected = numpy.array([[3.0, 4.0], [3.0, 4.0]])

    errors = _evaluate_error(measured, corrected, model)

    numpy.testing.assert_allclose(errors, [5.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# AffineTransform.__call__ overloads (ProbePosition variant)
# ---------------------------------------------------------------------------


def test_affine_transform_probe_position_overload_preserves_index() -> None:
    """The ProbePosition overload returns a new ProbePosition with the same index."""
    transform = AffineTransform(2.0, 0.5, 1.0, -0.5, 3.0, -2.0)
    position = ProbePosition(index=7, coordinate_x_m=4.0, coordinate_y_m=6.0)

    transformed = transform(position)

    assert transformed.index == 7
    assert transformed.coordinate_x_m == pytest.approx(2 * 4.0 + 0.5 * 6.0 + 1.0)
    assert transformed.coordinate_y_m == pytest.approx(-0.5 * 4.0 + 3.0 * 6.0 - 2.0)


# ---------------------------------------------------------------------------
# AffineTransformEstimator: model-layer wrapper validation + delegation
# ---------------------------------------------------------------------------


def _make_repo(items: dict[int, ProbePositionSequence]) -> ProbePositionsRepository:
    """Stub repository where repository[idx].get_probe_positions() returns the mapped sequence."""
    repo = MagicMock()

    def get_item(idx: int) -> MagicMock:
        item = MagicMock()
        item.get_probe_positions.return_value = items[idx]
        return item

    repo.__getitem__.side_effect = get_item
    return cast(ProbePositionsRepository, repo)


def _make_settings(
    num_iterations: int, threshold: float, min_inliers: int
) -> AffineTransformEstimatorSettings:
    settings = MagicMock()
    settings.num_shuffles.get_value.return_value = num_iterations
    settings.inlier_threshold.get_value.return_value = threshold
    settings.min_inliers.get_value.return_value = min_inliers
    return cast(AffineTransformEstimatorSettings, settings)


def test_estimator_delegates_to_api() -> None:
    """The wrapper resolves product indexes via the repository and delegates to the api function."""
    rng = numpy.random.default_rng(0)
    truth = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    measured = rng.uniform(-1.0, 1.0, size=(30, 2))
    corrected = truth(measured)

    repo = _make_repo(
        {0: _sequence_from_xy(measured), 1: _sequence_from_xy(corrected)},
    )
    settings = _make_settings(num_iterations=50, threshold=0.01, min_inliers=10)
    estimator = AffineTransformEstimator(
        rng=numpy.random.default_rng(1),
        settings=settings,
        repository=repo,
    )

    result = estimator.estimate(measured_product_indexes=[0], corrected_product_indexes=[1])

    numpy.testing.assert_allclose(_params(result), _params(truth), atol=1e-6)


def test_estimator_rejects_duplicate_indexes() -> None:
    estimator = AffineTransformEstimator(
        rng=numpy.random.default_rng(0),
        settings=_make_settings(num_iterations=10, threshold=0.1, min_inliers=3),
        repository=_make_repo({}),
    )

    with pytest.raises(ValueError, match='duplicated measured'):
        estimator.estimate(measured_product_indexes=[0, 0], corrected_product_indexes=[1])

    with pytest.raises(ValueError, match='duplicated corrected'):
        estimator.estimate(measured_product_indexes=[0], corrected_product_indexes=[1, 1])


def test_estimator_rejects_overlapping_index_sets() -> None:
    estimator = AffineTransformEstimator(
        rng=numpy.random.default_rng(0),
        settings=_make_settings(num_iterations=10, threshold=0.1, min_inliers=3),
        repository=_make_repo({}),
    )

    with pytest.raises(ValueError, match='appears in corrected and measured'):
        estimator.estimate(measured_product_indexes=[0, 1], corrected_product_indexes=[1, 2])
