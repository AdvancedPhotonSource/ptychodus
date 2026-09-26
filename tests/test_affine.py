from dataclasses import replace
from typing import cast
from unittest.mock import MagicMock
import math

import numpy
import pytest

from ptychodus.api.affine import estimate_affine_transform, transform_product
from ptychodus.api.affine import AffineTransform, AffineTransformComponents
from ptychodus.api.object import Object, ObjectGeometry, compute_object_geometry
from ptychodus.api.probe import ProbeGeometry, ProbeSequence
from ptychodus.api.typing import ComplexArrayType, RealArrayType
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.model.analysis.affine import AffineTransformEstimator
from ptychodus.model.analysis.settings import AffineTransformEstimatorSettings
from ptychodus.model.product import ProbePositionsRepository


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _params(t: AffineTransform) -> tuple[float, float, float, float, float, float]:
    return (t.a00, t.a01, t.a02, t.a10, t.a11, t.a12)


def _transform_xy(transform: AffineTransform, arr: numpy.ndarray) -> numpy.ndarray:
    """Apply ``transform`` to an (N, 2) array packed (x, y), returning the same packing."""
    x_m, y_m = transform.transform_coordinates(arr[:, 0], arr[:, 1])
    return numpy.column_stack((x_m, y_m))


def _sequence_from_xy(arr: numpy.ndarray) -> ProbePositionSequence:
    """Build a ProbePositionSequence from an (N, 2) array packed (x, y)."""
    return ProbePositionSequence(
        [
            ProbePosition(index=i, x_m=float(arr[i, 0]), y_m=float(arr[i, 1]))
            for i in range(arr.shape[0])
        ]
    )


# ---------------------------------------------------------------------------
# estimate_affine_transform: the robust coarse pass
# ---------------------------------------------------------------------------


def test_coarse_pass_recovers_transform_with_outliers() -> None:
    """With 20% gross outliers, the consensus pass still pins the affine within tolerance."""
    rng = numpy.random.default_rng(2026)
    n = 100
    truth = AffineTransform(1.05, 0.02, 1e-5, -0.03, 0.98, -2e-5)

    measured = rng.uniform(-1e-4, 1e-4, size=(n, 2))
    corrected = _transform_xy(truth, measured)

    n_outliers = 20
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    corrected[outlier_idx] += rng.uniform(-1e-3, 1e-3, size=(n_outliers, 2))

    result = estimate_affine_transform(
        [(_sequence_from_xy(measured), _sequence_from_xy(corrected))],
        num_iterations=200,
        inlier_threshold_m=1e-6,
        min_inliers=50,
        rng=numpy.random.default_rng(7),
    )

    numpy.testing.assert_allclose(
        [
            result.linear_transform.a00,
            result.linear_transform.a01,
            result.linear_transform.a10,
            result.linear_transform.a11,
        ],
        [truth.a00, truth.a01, truth.a10, truth.a11],
        atol=5e-3,
    )
    # Translation: scaled by the scan extent (~1e-4), so allow proportional tolerance.
    assert abs(result.pairs[0].translation_x_m - truth.a02) < 1e-6
    assert abs(result.pairs[0].translation_y_m - truth.a12) < 1e-6


def test_coarse_pass_runs_per_pair() -> None:
    """Two pairs sharing one linear part are each screened, then refined together."""
    rng = numpy.random.default_rng(99)
    truth = AffineTransform(1.2, 0.0, 5.0, 0.0, 1.2, -3.0)

    measured_a = rng.uniform(-1.0, 1.0, size=(15, 2))
    measured_b = rng.uniform(-1.0, 1.0, size=(15, 2))

    result = estimate_affine_transform(
        [
            (_sequence_from_xy(measured_a), _sequence_from_xy(_transform_xy(truth, measured_a))),
            (_sequence_from_xy(measured_b), _sequence_from_xy(_transform_xy(truth, measured_b))),
        ],
        num_iterations=100,
        inlier_threshold_m=1e-6,
        min_inliers=10,
        rng=numpy.random.default_rng(0),
    )

    numpy.testing.assert_allclose(
        _params(result.linear_transform),
        (truth.a00, truth.a01, 0.0, truth.a10, truth.a11, 0.0),
        atol=1e-6,
    )

    for pair in result.pairs:
        assert pair.num_points == 15
        assert (pair.translation_x_m, pair.translation_y_m) == pytest.approx((5.0, -3.0))


def test_coarse_pass_too_few_points_raises() -> None:
    """Need at least 3 points per pair to estimate an affine."""
    rng = numpy.random.default_rng(5)
    measured = rng.uniform(-1.0, 1.0, size=(2, 2))
    corrected = rng.uniform(-1.0, 1.0, size=(2, 2))

    with pytest.raises(ValueError, match='at least 3'):
        estimate_affine_transform(
            [(_sequence_from_xy(measured), _sequence_from_xy(corrected))],
            rng=numpy.random.default_rng(0),
        )


def test_coarse_pass_no_inliers_raises() -> None:
    """If the consensus pass never reaches min_inliers within threshold, raise."""
    rng = numpy.random.default_rng(3)
    measured = rng.uniform(-1.0, 1.0, size=(20, 2))
    # Garbage correspondences guarantee large per-point residuals.
    corrected = rng.uniform(-1.0, 1.0, size=(20, 2))

    with pytest.raises(RuntimeError, match='at least 15 inlier'):
        estimate_affine_transform(
            [(_sequence_from_xy(measured), _sequence_from_xy(corrected))],
            num_iterations=20,
            inlier_threshold_m=1e-10,
            min_inliers=15,
            rng=numpy.random.default_rng(0),
        )


def test_coarse_pass_default_rng_runs() -> None:
    """Smoke test that the rng=None default path works (auto-creates a generator)."""
    truth = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # identity
    rng = numpy.random.default_rng(6)
    measured = rng.uniform(-1.0, 1.0, size=(30, 2))

    result = estimate_affine_transform(
        [(_sequence_from_xy(measured), _sequence_from_xy(_transform_xy(truth, measured)))],
        num_iterations=50,
        inlier_threshold_m=1e-6,
        min_inliers=10,
    )

    numpy.testing.assert_allclose(_params(result.linear_transform), _params(truth), atol=1e-6)


def test_coarse_pass_recovers_transform_from_clean_data() -> None:
    """Without outliers the fit should be exact to within roundoff."""
    truth = AffineTransform(1.05, 0.02, 1e-5, -0.03, 0.98, -2e-5)
    rng = numpy.random.default_rng(1)
    measured = rng.uniform(-1e-4, 1e-4, size=(40, 2))

    result = estimate_affine_transform(
        [(_sequence_from_xy(measured), _sequence_from_xy(_transform_xy(truth, measured)))],
        num_iterations=100,
        inlier_threshold_m=1e-9,
        min_inliers=10,
        rng=numpy.random.default_rng(0),
    )

    numpy.testing.assert_allclose(
        _params(result.linear_transform),
        (truth.a00, truth.a01, 0.0, truth.a10, truth.a11, 0.0),
        atol=1e-12,
    )
    numpy.testing.assert_allclose(
        (result.pairs[0].translation_x_m, result.pairs[0].translation_y_m),
        (truth.a02, truth.a12),
        atol=1e-12,
    )


@pytest.mark.parametrize('half_extent_m', [1.0e-4, 1.0e-5, 2.5e-6, 1.0e-7])
def test_coarse_pass_is_scale_free(half_extent_m: float) -> None:
    """A clean scan fits whatever its physical size.

    The degeneracy screen compares a cross product -- an area -- so an absolute tolerance would
    make the verdict a function of the scan's extent: a five-micron scan would have every minimal
    sample rejected and the search would end empty.
    """
    truth = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    rng = numpy.random.default_rng(11)
    measured = rng.uniform(-half_extent_m, half_extent_m, size=(40, 2))

    result = estimate_affine_transform(
        [(_sequence_from_xy(measured), _sequence_from_xy(_transform_xy(truth, measured)))],
        num_iterations=100,
        inlier_threshold_m=half_extent_m * 1.0e-3,
        min_inliers=10,
        rng=numpy.random.default_rng(0),
    )

    numpy.testing.assert_allclose(
        _components(result.components), _components(_CALIBRATION_COMPONENTS), atol=1e-9
    )


def test_coarse_pass_tolerates_collinear_subsets() -> None:
    """A raster grid is full of collinear triples and still fits every one of its points."""
    truth = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    uncorrected = _grid_scan(10, 1.0e-6, (0.0, 0.0))

    result = estimate_affine_transform(
        [(uncorrected, _apply(uncorrected, truth))],
        num_iterations=100,
        rng=numpy.random.default_rng(0),
    )

    assert result.pairs[0].num_inliers == 100
    numpy.testing.assert_allclose(
        _components(result.components), _components(_CALIBRATION_COMPONENTS), atol=1e-11
    )


def test_coarse_pass_tolerates_two_parallel_rows() -> None:
    """Two rows still span the plane, so the linear part is determined."""
    truth = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    uncorrected = ProbePositionSequence(
        [ProbePosition(index=i, x_m=(i % 50) * 1.0e-6, y_m=(i // 50) * 1.0e-6) for i in range(100)]
    )

    result = estimate_affine_transform(
        [(uncorrected, _apply(uncorrected, truth))],
        num_iterations=100,
        rng=numpy.random.default_rng(0),
    )

    assert result.pairs[0].num_inliers == 100


def test_coarse_pass_rejects_a_one_dimensional_scan() -> None:
    """A line scan cannot determine a 2-D linear part, and is named as such rather than unlucky."""
    t = numpy.linspace(0.0, 1e-4, 20)
    collinear = numpy.stack([t, 2.0 * t + 1e-5], axis=1)

    with pytest.raises(ValueError, match='one-dimensional'):
        estimate_affine_transform(
            [(_sequence_from_xy(collinear), _sequence_from_xy(collinear.copy()))],
            num_iterations=50,
            inlier_threshold_m=1e-9,
            min_inliers=5,
            rng=numpy.random.default_rng(0),
        )


def test_coarse_pass_rejects_coincident_input() -> None:
    """Coincident points carry no geometry at all, and are reported as bad input."""
    coincident = numpy.zeros((10, 2))

    with pytest.raises(ValueError, match='one-dimensional'):
        estimate_affine_transform(
            [(_sequence_from_xy(coincident), _sequence_from_xy(coincident.copy()))],
            num_iterations=50,
            inlier_threshold_m=1e-9,
            min_inliers=5,
            rng=numpy.random.default_rng(0),
        )


def test_coarse_pass_honors_min_inliers() -> None:
    """A model supported by fewer than min_inliers points is refused even when it fits perfectly."""
    truth = AffineTransform(1.05, 0.02, 1e-5, -0.03, 0.98, -2e-5)
    rng = numpy.random.default_rng(1)
    measured = rng.uniform(-1e-4, 1e-4, size=(10, 2))

    with pytest.raises(RuntimeError, match='at least 50 inlier'):
        estimate_affine_transform(
            [(_sequence_from_xy(measured), _sequence_from_xy(_transform_xy(truth, measured)))],
            num_iterations=50,
            inlier_threshold_m=1e-9,
            min_inliers=50,
            rng=numpy.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# AffineTransform.__call__ overloads (ProbePosition variant)
# ---------------------------------------------------------------------------


def test_affine_transform_probe_position_overload_preserves_index() -> None:
    """The ProbePosition overload returns a new ProbePosition with the same index."""
    transform = AffineTransform(2.0, 0.5, 1.0, -0.5, 3.0, -2.0)
    position = ProbePosition(index=7, x_m=4.0, y_m=6.0)

    transformed = transform(position)

    assert transformed.index == 7
    assert transformed.x_m == pytest.approx(2 * 4.0 + 0.5 * 6.0 + 1.0)
    assert transformed.y_m == pytest.approx(-0.5 * 4.0 + 3.0 * 6.0 - 2.0)


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
    settings.num_iterations.get_value.return_value = num_iterations
    settings.inlier_threshold_m.get_value.return_value = threshold
    settings.min_inliers.get_value.return_value = min_inliers
    return cast(AffineTransformEstimatorSettings, settings)


def test_estimator_delegates_to_api() -> None:
    """The wrapper resolves product indexes via the repository and delegates to the api function."""
    rng = numpy.random.default_rng(0)
    truth = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    measured = rng.uniform(-1.0, 1.0, size=(30, 2))
    corrected = _transform_xy(truth, measured)

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

    numpy.testing.assert_allclose(_params(result.linear_transform), _params(truth), atol=1e-6)


def test_estimator_rejects_mismatched_index_counts() -> None:
    """The two index sequences correspond elementwise, so they must be the same length."""
    estimator = AffineTransformEstimator(
        rng=numpy.random.default_rng(0),
        settings=_make_settings(num_iterations=10, threshold=0.1, min_inliers=3),
        repository=_make_repo({}),
    )

    with pytest.raises(ValueError, match='at the same offset'):
        estimator.estimate(measured_product_indexes=[0, 1], corrected_product_indexes=[2])


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


# ---------------------------------------------------------------------------
# AffineTransformComponents: decomposition and composition
# ---------------------------------------------------------------------------


def _components(c: AffineTransformComponents) -> tuple[float, float, float, float]:
    return (c.scale, c.asymmetry, c.rotation_rad, c.shear_rad)


def test_decompose_identity() -> None:
    """The identity transform has unit scale and no asymmetry, rotation or shear."""
    assert _components(AffineTransform.create_identity().decompose()) == (1.0, 0.0, 0.0, 0.0)


def test_decompose_pure_scale() -> None:
    """An isotropic magnification shows up entirely in the scale component."""
    components = AffineTransform(1.01, 0.0, 5.0, 0.0, 1.01, -3.0).decompose()

    numpy.testing.assert_allclose(_components(components), (1.01, 0.0, 0.0, 0.0), atol=1e-14)


def test_decompose_pure_rotation() -> None:
    """A rotation matrix decomposes to unit scale and that rotation angle."""
    angle_rad = 0.3
    transform = AffineTransform(
        math.cos(angle_rad),
        math.sin(angle_rad),
        0.0,
        -math.sin(angle_rad),
        math.cos(angle_rad),
        0.0,
    )

    numpy.testing.assert_allclose(
        _components(transform.decompose()), (1.0, 0.0, angle_rad, 0.0), atol=1e-14
    )


def test_decompose_pure_asymmetry() -> None:
    """Unequal magnification of x and y is reported as asymmetry about their mean scale."""
    components = AffineTransform(1.1, 0.0, 0.0, 0.0, 0.9, 0.0).decompose()

    numpy.testing.assert_allclose(_components(components), (1.0, 0.2, 0.0, 0.0), atol=1e-14)


def test_decompose_pure_shear() -> None:
    """A shear of the second axis along the first is reported as the shear angle."""
    angle_rad = 0.2
    components = AffineTransform(1.0, 0.0, 0.0, math.tan(angle_rad), 1.0, 0.0).decompose()

    numpy.testing.assert_allclose(_components(components), (1.0, 0.0, 0.0, angle_rad), atol=1e-14)


def test_decompose_ignores_translation() -> None:
    """Only the linear part is decomposed; the translation columns do not participate."""
    linear = (1.03, 0.02, -0.01, 0.98)
    without = AffineTransform(linear[0], linear[1], 0.0, linear[2], linear[3], 0.0)
    with_shift = AffineTransform(linear[0], linear[1], 7.0, linear[2], linear[3], -4.0)

    assert without.decompose() == with_shift.decompose()


def test_from_components_inverts_decompose() -> None:
    """Composing the decomposed components rebuilds the original linear part exactly."""
    truth = AffineTransform(1.0031, 0.0021, 1.0e-8, -0.0030, 1.0033, -2.0e-8)

    rebuilt = AffineTransform.from_components(
        truth.decompose(), translation_x_m=truth.a02, translation_y_m=truth.a12
    )

    numpy.testing.assert_allclose(_params(rebuilt), _params(truth), rtol=1e-12, atol=1e-15)


def test_decompose_round_trips_over_random_transforms() -> None:
    """Decompose then compose reproduces the linear part to machine precision.

    The parameterization admits two shear branches, so the recovered components need not equal
    the ones a matrix was built from; the matrix they generate must still match.
    """
    rng = numpy.random.default_rng(11)

    for _ in range(500):
        truth = AffineTransform.from_components(
            AffineTransformComponents(
                scale=float(rng.uniform(0.2, 5.0)),
                asymmetry=float(rng.uniform(-1.5, 1.5)),
                rotation_rad=float(rng.uniform(-1.4, 1.4)),
                shear_rad=float(rng.uniform(-1.2, 1.2)),
            )
        )

        rebuilt = AffineTransform.from_components(truth.decompose())

        numpy.testing.assert_allclose(_params(rebuilt), _params(truth), rtol=1e-11, atol=1e-12)


def test_decompose_selects_minimum_asymmetry_branch() -> None:
    """Of the two shear branches that generate a matrix, the one nearer the identity is chosen."""
    truth = AffineTransformComponents(
        scale=1.5044859047269257,
        asymmetry=1.485138155100452,
        rotation_rad=0.14719781247008235,
        shear_rad=0.47757792422135004,
    )
    transform = AffineTransform.from_components(truth)

    recovered = transform.decompose()

    assert abs(recovered.asymmetry) < abs(truth.asymmetry)
    numpy.testing.assert_allclose(
        _params(AffineTransform.from_components(recovered)), _params(transform), atol=1e-12
    )


def test_decompose_quadratic_eps_forces_the_linear_branch() -> None:
    """The guard on the quadratic coefficient is a caller-visible tolerance.

    At the default it is small enough that a genuine quadratic is solved as one; raised to unity
    every quadratic counts as roundoff, the single linear root is taken instead, and the
    components no longer regenerate the matrix.
    """
    transform = AffineTransform.from_components(
        AffineTransformComponents(scale=1.2, asymmetry=0.3, rotation_rad=0.2, shear_rad=0.4)
    )

    numpy.testing.assert_allclose(
        _params(AffineTransform.from_components(transform.decompose())),
        _params(transform),
        rtol=1e-12,
        atol=1e-14,
    )

    degraded = AffineTransform.from_components(transform.decompose(quadratic_eps=1.0))

    assert not numpy.allclose(_params(degraded), _params(transform), rtol=1e-6, atol=1e-9)


def test_decompose_rejects_singular_linear_part() -> None:
    """A singular linear part has no scale/rotation factorization."""
    with pytest.raises(ValueError, match='determinant'):
        AffineTransform(1.0, 2.0, 0.0, 2.0, 4.0, 0.0).decompose()


def test_decompose_rejects_reflection() -> None:
    """An orientation-reversing transform is refused rather than returned as a huge asymmetry."""
    with pytest.raises(ValueError, match='orientation-preserving'):
        AffineTransform(1.0, 0.0, 0.0, 0.0, -1.0, 0.0).decompose()


# ---------------------------------------------------------------------------
# estimate_affine_transform: joint multi-pair refinement
# ---------------------------------------------------------------------------


def _grid_scan(
    n_side: int, step_m: float, origin: tuple[float, float], start_index: int = 0
) -> ProbePositionSequence:
    """Raster grid of ``n_side`` by ``n_side`` positions with consecutive scan indexes."""
    return ProbePositionSequence(
        [
            ProbePosition(
                index=start_index + iy * n_side + ix,
                x_m=origin[0] + ix * step_m,
                y_m=origin[1] + iy * step_m,
            )
            for iy in range(n_side)
            for ix in range(n_side)
        ]
    )


def _apply(
    positions: ProbePositionSequence,
    transform: AffineTransform,
    noise_m: float = 0.0,
    rng: numpy.random.Generator | None = None,
) -> ProbePositionSequence:
    """Map every position through ``transform``, optionally adding Gaussian noise."""
    generator = numpy.random.default_rng(0) if rng is None else rng
    moved = []

    for point in positions:
        image = transform(point)
        moved.append(
            ProbePosition(
                index=point.index,
                x_m=image.x_m + noise_m * float(generator.normal()),
                y_m=image.y_m + noise_m * float(generator.normal()),
            )
        )

    return ProbePositionSequence(moved)


_CALIBRATION_COMPONENTS = AffineTransformComponents(
    scale=1.0031,
    asymmetry=-4.0e-4,
    rotation_rad=math.radians(0.12),
    shear_rad=math.radians(-0.05),
)


def test_joint_fit_recovers_transform_from_one_pair() -> None:
    """A single noiseless pair pins down the shared linear part and its translation."""
    linear = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    full = replace(linear, a02=1.0e-8, a12=-2.0e-8)
    uncorrected = _grid_scan(10, 1.0e-6, (0.0, 0.0))

    result = estimate_affine_transform([(uncorrected, _apply(uncorrected, full))])

    numpy.testing.assert_allclose(
        _components(result.components), _components(_CALIBRATION_COMPONENTS), atol=1e-12
    )
    numpy.testing.assert_allclose(
        _params(result.linear_transform),
        (full.a00, full.a01, 0.0, full.a10, full.a11, 0.0),
        atol=1e-12,
    )
    numpy.testing.assert_allclose(
        (result.pairs[0].translation_x_m, result.pairs[0].translation_y_m),
        (full.a02, full.a12),
        atol=1e-12,
    )


def test_joint_fit_shares_linear_part_across_independent_origins() -> None:
    """Scans at different stage origins with different translations share one linear part.

    A single pooled fit with one translation would absorb the origin differences into the linear
    part; this checks the per-pair translation keeps them out of it.
    """
    linear = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    origins = [(0.0, 0.0), (5.0e-4, -3.0e-4), (-9.0e-4, 8.0e-4)]
    shifts = [(1.0e-8, -2.0e-8), (-7.0e-8, 4.0e-8), (2.0e-9, 9.0e-9)]
    pairs = []

    for index, (origin, shift) in enumerate(zip(origins, shifts)):
        uncorrected = _grid_scan(8, 1.0e-6, origin, start_index=1000 * index)
        full = replace(linear, a02=shift[0], a12=shift[1])
        pairs.append((uncorrected, _apply(uncorrected, full)))

    result = estimate_affine_transform(pairs)

    numpy.testing.assert_allclose(
        _components(result.components), _components(_CALIBRATION_COMPONENTS), atol=1e-11
    )

    for index, shift in enumerate(shifts):
        assert result.pairs[index].num_points == 64
        assert result.pairs[index].num_inliers == 64
        numpy.testing.assert_allclose(
            (result.pairs[index].translation_x_m, result.pairs[index].translation_y_m),
            shift,
            atol=1e-15,
        )


def test_joint_fit_pairs_by_scan_index_not_array_order() -> None:
    """Shuffling the corrected sequence changes nothing, because matching is by scan index."""
    full = replace(AffineTransform.from_components(_CALIBRATION_COMPONENTS), a02=3.0e-9)
    uncorrected = _grid_scan(6, 1.0e-6, (0.0, 0.0))
    corrected = _apply(uncorrected, full)
    shuffled = ProbePositionSequence(list(corrected)[::-1])

    ordered = estimate_affine_transform([(uncorrected, corrected)], rng=numpy.random.default_rng(0))
    reversed_ = estimate_affine_transform(
        [(uncorrected, shuffled)], rng=numpy.random.default_rng(0)
    )

    assert _components(ordered.components) == _components(reversed_.components)


def test_joint_fit_uses_only_shared_scan_indexes() -> None:
    """A corrected set that lost scan points still contributes every point it kept."""
    full = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    uncorrected = _grid_scan(6, 1.0e-6, (0.0, 0.0))
    kept = ProbePositionSequence([p for i, p in enumerate(_apply(uncorrected, full)) if i % 3])

    result = estimate_affine_transform([(uncorrected, kept)])

    assert result.pairs[0].num_points == 24
    numpy.testing.assert_allclose(
        _components(result.components), _components(_CALIBRATION_COMPONENTS), atol=1e-11
    )


def test_joint_fit_rejects_outliers() -> None:
    """Sigma clipping keeps a few badly refined positions from dragging the scale."""
    full = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    uncorrected = _grid_scan(10, 1.0e-6, (0.0, 0.0))
    corrupted = list(_apply(uncorrected, full))

    for row in (3, 40, 77):
        point = corrupted[row]
        corrupted[row] = ProbePosition(point.index, point.x_m + 5.0e-7, point.y_m - 4.0e-7)

    pair = (uncorrected, ProbePositionSequence(corrupted))
    # The coarse pass is deliberately slack enough to admit the corrupted points, so that what
    # differs between the two fits is the trim and not the consensus threshold.
    options = {'inlier_threshold_m': 1.0e-5, 'rng': numpy.random.default_rng(0)}
    trimmed = estimate_affine_transform([pair], **options)  # type: ignore[arg-type]
    untrimmed = estimate_affine_transform(  # type: ignore[arg-type]
        [pair], outlier_sigma=math.inf, **options
    )

    assert trimmed.pairs[0].num_inliers == 97
    assert untrimmed.pairs[0].num_inliers == 100
    assert abs(trimmed.components.scale - _CALIBRATION_COMPONENTS.scale) < abs(
        untrimmed.components.scale - _CALIBRATION_COMPONENTS.scale
    )


def test_joint_fit_reports_residuals() -> None:
    """Residuals are Euclidean distances in meters, near zero for a noiseless fit."""
    full = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    uncorrected = _grid_scan(6, 1.0e-6, (0.0, 0.0))

    result = estimate_affine_transform([(uncorrected, _apply(uncorrected, full))])

    assert result.rms_residual_m < 1e-18
    assert result.pairs[0].rms_residual_m < 1e-18
    assert 'scale=1.003100' in str(result)


def test_joint_fit_requires_a_pair() -> None:
    """An empty pair list is refused rather than returning the identity."""
    with pytest.raises(ValueError, match='At least one'):
        estimate_affine_transform([])


def test_joint_fit_requires_shared_indexes() -> None:
    """Two sequences with disjoint scan indexes cannot be paired."""
    left = _grid_scan(4, 1.0e-6, (0.0, 0.0), start_index=0)
    right = _grid_scan(4, 1.0e-6, (0.0, 0.0), start_index=500)

    with pytest.raises(ValueError, match='shares no scan indexes'):
        estimate_affine_transform([(left, right)])


def test_joint_fit_pools_across_pairs_better_than_any_pair_alone() -> None:
    """Three noisy scans sharing one linear part recover it better together than separately."""
    truth = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    origins = [(0.0, 0.0), (5.0e-4, -3.0e-4), (-9.0e-4, 8.0e-4)]
    rng = numpy.random.default_rng(31)
    pairs = [
        (
            scan := _grid_scan(6, 1.0e-6, origin, start_index=1000 * index),
            _apply(scan, truth, noise_m=2.0e-9, rng=rng),
        )
        for index, origin in enumerate(origins)
    ]

    def error(result: object) -> float:
        components = getattr(result, 'components')
        return float(
            numpy.linalg.norm(
                numpy.subtract(_components(components), _components(_CALIBRATION_COMPONENTS))
            )
        )

    joint = error(estimate_affine_transform(pairs, rng=numpy.random.default_rng(0)))
    alone = [
        error(estimate_affine_transform([pair], rng=numpy.random.default_rng(0))) for pair in pairs
    ]

    assert joint < min(alone)


def test_joint_fit_shared_part_is_invariant_to_translating_one_pair() -> None:
    """Moving a whole scan in lab coordinates leaves the shared linear part untouched.

    This is the per-pair centering stated directly: the translation is profiled out rather than
    absorbed into the linear part.
    """
    truth = AffineTransform.from_components(_CALIBRATION_COMPONENTS)
    pairs = [
        (scan := _grid_scan(6, 1.0e-6, origin, start_index=1000 * index), _apply(scan, truth))
        for index, origin in enumerate([(0.0, 0.0), (5.0e-4, -3.0e-4)])
    ]
    shifted = ProbePositionSequence(
        [ProbePosition(index=p.index, x_m=p.x_m + 4.0e-7, y_m=p.y_m - 6.0e-7) for p in pairs[1][1]]
    )
    moved = [pairs[0], (pairs[1][0], shifted)]

    base = estimate_affine_transform(pairs, rng=numpy.random.default_rng(0))
    shifted_fit = estimate_affine_transform(moved, rng=numpy.random.default_rng(0))

    numpy.testing.assert_allclose(
        _components(shifted_fit.components), _components(base.components), atol=1e-12
    )
    numpy.testing.assert_allclose(
        (shifted_fit.pairs[1].translation_x_m, shifted_fit.pairs[1].translation_y_m),
        (base.pairs[1].translation_x_m + 4.0e-7, base.pairs[1].translation_y_m - 6.0e-7),
        atol=1e-15,
    )


def test_joint_fit_rejects_collinear_positions() -> None:
    """Points on a line do not determine the linear part, even in quantity."""
    positions = ProbePositionSequence(
        [ProbePosition(index=i, x_m=i * 1.0e-6, y_m=0.0) for i in range(20)]
    )
    corrected = _apply(positions, AffineTransform.from_components(_CALIBRATION_COMPONENTS))

    with pytest.raises(ValueError, match='one-dimensional'):
        estimate_affine_transform([(positions, corrected)])


def test_joint_fit_rejects_too_few_points() -> None:
    """A full affine needs at least three matched points per pair."""
    positions = ProbePositionSequence([ProbePosition(index=0, x_m=1.0e-6, y_m=2.0e-6)])

    with pytest.raises(ValueError, match='at least 3'):
        estimate_affine_transform([(positions, positions)])


# ---------------------------------------------------------------------------
# transform_product
# ---------------------------------------------------------------------------


def _probe_geometry() -> ProbeGeometry:
    return ProbeGeometry(width_px=8, height_px=8, pixel_width_m=1.0e-8, pixel_height_m=1.0e-8)


def _smooth_field(x_m: RealArrayType, y_m: RealArrayType) -> ComplexArrayType:
    """Band-limited complex field: 200 pixels per fringe, far below Nyquist.

    Bilinear resampling smooths content that reaches toward Nyquist badly -- 43% relative error at
    70% of Nyquist -- so a fixture meant to measure the coordinate bookkeeping rather than the
    interpolator has to stay well away from it.
    """
    envelope = numpy.exp(-(numpy.square(x_m / 4.0e-6) + numpy.square(y_m / 4.0e-6)))
    phase = 2.0 * numpy.pi * (x_m / 2.0e-6 + y_m / 3.0e-6)
    return (1.0 + 0.5 * envelope) * numpy.exp(1j * phase)


def _sample_field(geometry: ObjectGeometry) -> ComplexArrayType:
    coordinates = geometry.get_transverse_coordinates()
    x_m = geometry.center_x_m + coordinates.x_m
    y_m = geometry.center_y_m + coordinates.y_m
    return _smooth_field(x_m, y_m)[numpy.newaxis, ...]


def _make_product(
    positions: ProbePositionSequence,
    *,
    focus_object_distance_m: float = 0.0,
    padding_px: int = 0,
    object_array: ComplexArrayType | None = None,
) -> Product:
    """Product whose object is the canvas its own scan implies, optionally padded or replaced."""
    probe_geometry = _probe_geometry()
    geometry = compute_object_geometry(positions, probe_geometry, padding_px=padding_px)

    if object_array is None:
        object_array = _sample_field(geometry)

    return Product(
        metadata=ProductMetadata(
            name='test',
            comments='',
            detector_distance_m=2.335,
            probe_energy_eV=10000.0,
            probe_photon_count=1.0e9,
            exposure_time_s=0.1,
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=0.0,
            focus_object_distance_m=focus_object_distance_m,
        ),
        probe_positions=positions,
        probes=ProbeSequence(
            array=numpy.zeros((1, 1, 8, 8), dtype=complex),
            opr_weights=None,
            pixel_geometry=probe_geometry.get_pixel_geometry(),
        ),
        object_=Object(
            array=object_array,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        ),
        losses=[],
    )


def _corrected_probe_geometry(scale: float) -> ProbeGeometry:
    probe_geometry = _probe_geometry()
    return ProbeGeometry(
        width_px=probe_geometry.width_px,
        height_px=probe_geometry.height_px,
        pixel_width_m=probe_geometry.pixel_width_m / scale,
        pixel_height_m=probe_geometry.pixel_height_m / scale,
    )


def _scale_free(components: AffineTransformComponents) -> AffineTransform:
    return AffineTransform.from_components(replace(components, scale=1.0))


def test_transform_product_moves_scale_into_the_detector_distance() -> None:
    """The scale lands on the detector distance and is gone from the returned positions."""
    positions = _grid_scan(6, 1.0e-6, (0.0, 0.0))
    product = _make_product(positions)
    transform = AffineTransform.from_components(_CALIBRATION_COMPONENTS)

    corrected = transform_product(product, transform)

    assert corrected.metadata.detector_distance_m == pytest.approx(
        2.335 / _CALIBRATION_COMPONENTS.scale
    )
    refit = estimate_affine_transform(
        [(positions, corrected.probe_positions)], rng=numpy.random.default_rng(0)
    )
    assert refit.components.scale == pytest.approx(1.0, abs=1e-12)


def test_transform_product_keeps_the_scale_free_components() -> None:
    """Asymmetry, rotation and shear stay on the probe positions; the translation is dropped."""
    positions = _grid_scan(6, 1.0e-6, (0.0, 0.0))
    transform = AffineTransform.from_components(
        _CALIBRATION_COMPONENTS, translation_x_m=1.0e-8, translation_y_m=-2.0e-8
    )

    corrected = transform_product(_make_product(positions), transform)

    refit = estimate_affine_transform(
        [(positions, corrected.probe_positions)], rng=numpy.random.default_rng(0)
    )
    numpy.testing.assert_allclose(
        (refit.components.asymmetry, refit.components.rotation_rad, refit.components.shear_rad),
        (
            _CALIBRATION_COMPONENTS.asymmetry,
            _CALIBRATION_COMPONENTS.rotation_rad,
            _CALIBRATION_COMPONENTS.shear_rad,
        ),
        atol=1e-12,
    )
    numpy.testing.assert_allclose(
        (refit.pairs[0].translation_x_m, refit.pairs[0].translation_y_m), (0.0, 0.0), atol=1e-15
    )


def test_transform_product_preserves_indexes_and_photon_counts() -> None:
    """Transforming positions moves coordinates only; the index and photon count ride along."""
    positions = ProbePositionSequence(
        [
            ProbePosition(
                index=7 * i,
                x_m=1.0e-6 * (i % 4),
                y_m=1.0e-6 * (i // 4),
                probe_photon_count=2.0e6 + i,
            )
            for i in range(16)
        ]
    )
    transform = AffineTransform.from_components(_CALIBRATION_COMPONENTS)

    corrected = transform_product(_make_product(positions), transform)

    assert [p.index for p in corrected.probe_positions] == [p.index for p in positions]
    assert [p.probe_photon_count for p in corrected.probe_positions] == [
        p.probe_photon_count for p in positions
    ]


def test_transform_product_scales_both_pixel_geometries() -> None:
    """The far-field object pixel follows the detector distance, so both pitches divide by scale."""
    product = _make_product(_grid_scan(6, 1.0e-6, (0.0, 0.0)))
    scale = _CALIBRATION_COMPONENTS.scale

    corrected = transform_product(product, AffineTransform.from_components(_CALIBRATION_COMPONENTS))

    for pixel_geometry in (
        corrected.object_.get_pixel_geometry(),
        corrected.probes.get_geometry().get_pixel_geometry(),
    ):
        assert pixel_geometry.width_m == pytest.approx(1.0e-8 / scale)
        assert pixel_geometry.height_m == pytest.approx(1.0e-8 / scale)


def test_transform_product_reinterprets_the_probe_without_resampling_it() -> None:
    """Only the scan frame is distorted, so the probe array itself is left exactly alone."""
    product = _make_product(_grid_scan(6, 1.0e-6, (0.0, 0.0)))

    corrected = transform_product(product, AffineTransform.from_components(_CALIBRATION_COMPONENTS))

    numpy.testing.assert_array_equal(corrected.probes.get_array(), product.probes.get_array())


@pytest.mark.parametrize('padding_px', [0, 4])
def test_transform_product_object_geometry_follows_the_sizing_rule(padding_px: int) -> None:
    """The corrected canvas is the rule's answer for the corrected scan plus the input's own extra."""
    positions = _grid_scan(6, 1.0e-6, (0.0, 0.0))
    product = _make_product(positions, padding_px=padding_px)
    components = _CALIBRATION_COMPONENTS
    corrected_probe_geometry = _corrected_probe_geometry(components.scale)

    corrected = transform_product(product, AffineTransform.from_components(components))

    expected = compute_object_geometry(
        corrected.probe_positions, corrected_probe_geometry, padding_px=padding_px
    )
    geometry = corrected.object_.get_geometry()
    assert (geometry.width_px, geometry.height_px) == (expected.width_px, expected.height_px)
    assert geometry.pixel_width_m == pytest.approx(expected.pixel_width_m)
    assert geometry.pixel_height_m == pytest.approx(expected.pixel_height_m)
    assert corrected.object_.get_array().shape == (1, geometry.height_px, geometry.width_px)
    assert geometry.center_x_m == pytest.approx(expected.center_x_m, abs=1e-15)
    assert geometry.center_y_m == pytest.approx(expected.center_y_m, abs=1e-15)

    # The canvas still covers every corrected position with room for the probe around it.
    margin_x_m = corrected_probe_geometry.width_m / 2
    margin_y_m = corrected_probe_geometry.height_m / 2

    for position in corrected.probe_positions:
        assert geometry.minimum_x_m + margin_x_m <= position.x_m
        assert position.x_m <= geometry.minimum_x_m + geometry.width_m - margin_x_m
        assert geometry.minimum_y_m + margin_y_m <= position.y_m
        assert position.y_m <= geometry.minimum_y_m + geometry.height_m - margin_y_m


def test_transform_product_changes_the_object_shape_on_each_axis_independently() -> None:
    """A finer pixel needs more of them, and asymmetry and rotation act differently per axis."""
    product = _make_product(_grid_scan(6, 1.0e-6, (0.0, 0.0)))

    corrected = transform_product(product, AffineTransform.from_components(_CALIBRATION_COMPONENTS))

    before = product.object_.get_array().shape
    after = corrected.object_.get_array().shape
    assert after[1] > before[1]
    assert after[2] > before[2]
    assert after[1] != after[2]


@pytest.mark.parametrize('padding_px,oversize_px', [(0, 0), (4, 0), (0, 1024)])
def test_transform_product_identity_is_an_exact_no_op(padding_px: int, oversize_px: int) -> None:
    """An identity transform must not crop, pad, shift or smooth the object.

    The oversized case stands in for a warm start read from file, whose size is fixed by the file
    and is nowhere near the minimal canvas its scan implies; recomputing the canvas from the scan
    alone would crop it by hundreds of pixels.
    """
    positions = _grid_scan(6, 1.0e-6, (0.0, 0.0))
    geometry = compute_object_geometry(positions, _probe_geometry(), padding_px=padding_px)

    if oversize_px:
        geometry = replace(geometry, width_px=oversize_px, height_px=oversize_px)

    product = _make_product(positions, padding_px=padding_px, object_array=_sample_field(geometry))
    product = replace(
        product,
        object_=Object(
            array=product.object_.get_array(),
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        ),
    )

    corrected = transform_product(product, AffineTransform.create_identity())

    assert corrected.object_.get_geometry() == product.object_.get_geometry()
    numpy.testing.assert_allclose(
        corrected.object_.get_array(), product.object_.get_array(), atol=1e-12
    )
    assert corrected.metadata.detector_distance_m == pytest.approx(2.335)


def test_transform_product_resamples_the_object_onto_the_corrected_grid() -> None:
    """Every corrected pixel carries the source field evaluated at its preimage.

    The fixture is deliberately band-limited; the tolerance is what a single bilinear pass costs
    on content that smooth, and a sharp-edged object would fail it by an order of magnitude.
    """
    positions = _grid_scan(6, 1.0e-6, (0.0, 0.0))
    product = _make_product(positions)
    components = _CALIBRATION_COMPONENTS

    corrected = transform_product(product, AffineTransform.from_components(components))

    geometry = corrected.object_.get_geometry()
    coordinates = geometry.get_transverse_coordinates()
    x_m = geometry.center_x_m + coordinates.x_m
    y_m = geometry.center_y_m + coordinates.y_m
    scale_free = _scale_free(components)
    inverse = numpy.linalg.inv(
        numpy.array([[scale_free.a00, scale_free.a01], [scale_free.a10, scale_free.a11]])
    )
    expected = _smooth_field(
        inverse[0, 0] * x_m + inverse[0, 1] * y_m, inverse[1, 0] * x_m + inverse[1, 1] * y_m
    )

    # Crop the border, where target pixels map outside the source array and fill with zeros.
    interior = (slice(8, -8), slice(8, -8))
    actual = corrected.object_.get_array()[0][interior]

    assert actual.dtype == product.object_.get_array().dtype
    numpy.testing.assert_allclose(actual, expected[interior], atol=2e-3)


def test_transform_product_does_not_mutate_its_input() -> None:
    positions = _grid_scan(4, 1.0e-6, (0.0, 0.0))
    product = _make_product(positions)
    array = product.object_.get_array().copy()

    transform_product(product, AffineTransform.from_components(_CALIBRATION_COMPONENTS))

    assert product.metadata.detector_distance_m == 2.335
    assert product.probe_positions[0].x_m == 0.0
    numpy.testing.assert_array_equal(product.object_.get_array(), array)


def test_transform_product_rejects_near_field() -> None:
    """With a focusing optic the object pixel follows the magnification, not the distance."""
    product = _make_product(_grid_scan(4, 1.0e-6, (0.0, 0.0)), focus_object_distance_m=-0.01)

    with pytest.raises(ValueError, match='far-field'):
        transform_product(product, AffineTransform.from_components(_CALIBRATION_COMPONENTS))
