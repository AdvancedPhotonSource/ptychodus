"""Affine transforms of probe positions: the transform primitive, its physical decomposition,
the estimator that fits one to measured/corrected position pairs, and the product-level correction
that folds a fitted transform back into the experiment geometry."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
import logging
import math

import numpy
import skimage.transform
from scipy.ndimage import map_coordinates
from skimage.measure import ransac

from .geometry import PixelGeometry
from .object import Object, ObjectGeometry, compute_object_geometry
from .probe import ProbeGeometry, ProbeSequence
from .probe_positions import ProbePosition, ProbePositionSequence
from .product import Product
from .typing import BooleanArrayType, ComplexArrayType, IntegerArrayType, RealArrayType

__all__ = [
    'AffineFitResult',
    'AffinePairFit',
    'AffineTransform',
    'AffineTransformComponents',
    'estimate_affine_transform',
    'transform_probe_positions',
    'transform_product',
]

logger = logging.getLogger(__name__)

_AFFINE_MINIMUM_POINTS = 3  # 6 DOF / 2 equations per point
# Phi^-1(0.75): a Gaussian's median absolute deviation is this fraction of its standard deviation,
# so dividing a median absolute deviation by it rescales it to a standard-deviation estimate.
_MAD_TO_SIGMA = 0.6744897501960817


@dataclass(frozen=True)
class AffineTransformComponents:
    """Physical decomposition of the 2x2 linear part of an :class:`AffineTransform`.

    The linear part factors as ``M = scale * D * R * S`` with

    * ``D = diag(1 + asymmetry / 2, 1 - asymmetry / 2)`` -- unequal magnification of the two axes,
    * ``R = [[cos, sin], [-sin, cos]]`` evaluated at ``rotation_rad``,
    * ``S = [[1, 0], [tan(shear_rad), 1]]`` -- a shear of the second axis along the first.

    The factorization is expressed in the (x, y) frame that :class:`AffineTransform` acts in:
    column 0 / the first axis is x. ``scale`` is invariant under swapping the two axes, but
    ``asymmetry``, ``rotation_rad`` and ``shear_rad`` are not, so values quoted in a (y, x) frame
    do not transfer unchanged.
    """

    scale: float
    asymmetry: float
    rotation_rad: float
    shear_rad: float


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

    @classmethod
    def create_identity(cls) -> AffineTransform:
        """The identity transform."""
        return cls(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @classmethod
    def from_components(
        cls,
        components: AffineTransformComponents,
        *,
        translation_x_m: float = 0.0,
        translation_y_m: float = 0.0,
    ) -> AffineTransform:
        """Rebuild a transform from its physical components and an explicit translation.

        Inverse of :meth:`decompose` up to the translation, which the decomposition does not see.
        """
        k1 = components.scale * (1.0 + components.asymmetry / 2.0)
        k2 = components.scale * (1.0 - components.asymmetry / 2.0)
        cos_rotation = math.cos(components.rotation_rad)
        sin_rotation = math.sin(components.rotation_rad)
        tan_shear = math.tan(components.shear_rad)
        return cls(
            k1 * (cos_rotation + tan_shear * sin_rotation),
            k1 * sin_rotation,
            translation_x_m,
            k2 * (tan_shear * cos_rotation - sin_rotation),
            k2 * cos_rotation,
            translation_y_m,
        )

    def decompose(self, *, quadratic_eps: float = 1.0e-12) -> AffineTransformComponents:
        """Factor the linear part into scale, asymmetry, rotation and shear.

        Solved in closed form, so the result reproduces the linear part to machine precision.
        The factorization admits two shear branches that generate the same matrix; the branch with
        the smaller ``|asymmetry|`` is returned, which is the one continuous with the identity.

        Args:
            quadratic_eps: Fraction of the largest coefficient at or below which the quadratic
                coefficient of the shear equation counts as roundoff and the equation is solved as
                a linear one, giving the single shear branch. Raising it treats more matrices as
                shear-free in the quadratic term; lowering it keeps solving a quadratic whose
                leading coefficient is comparable to the cancellation error in forming it.

        Raises:
            ValueError: if the linear part is singular or orientation-reversing. A positive
                determinant is what makes ``scale`` positive and ``|asymmetry|`` less than 2.
        """
        determinant = self.a00 * self.a11 - self.a01 * self.a10

        if not math.isfinite(determinant) or determinant <= 0.0:
            raise ValueError(
                'Affine decomposition needs a nonsingular, orientation-preserving linear part; '
                f'got determinant {determinant}.'
            )

        # tan(shear) satisfies quadratic * t**2 + linear * t + constant == 0, obtained by equating
        # the two expressions the factorization gives for tan(rotation).
        quadratic = self.a01 * self.a11
        linear = -(self.a01 * self.a10 + self.a00 * self.a11)
        constant = self.a00 * self.a10 + self.a01 * self.a11
        largest = max(abs(quadratic), abs(linear), abs(constant))

        if abs(quadratic) <= quadratic_eps * largest:
            tangents = [0.0] if linear == 0.0 else [-constant / linear]
        else:
            discriminant = math.sqrt(max(linear * linear - 4.0 * quadratic * constant, 0.0))
            # Cancellation-free pairing: form the root whose terms have matching signs, then get
            # the other from the product of the roots.
            helper = -0.5 * (linear + math.copysign(discriminant, linear or 1.0))
            tangents = [0.0] if helper == 0.0 else [helper / quadratic, constant / helper]

        candidates = [self._components_for_shear_tangent(t) for t in tangents]
        return min(candidates, key=lambda c: abs(c.asymmetry))

    def _components_for_shear_tangent(self, tan_shear: float) -> AffineTransformComponents:
        """Components implied by one root of the shear equation."""
        # With k1 = scale * (1 + asymmetry / 2) and k2 = scale * (1 - asymmetry / 2), the
        # factorization gives k1 * cos = a00 - t * a01, k1 * sin = a01, k2 * cos = a11 and
        # k2 * sin = t * a11 - a10.
        k1_cos = self.a00 - tan_shear * self.a01
        k1_sin = self.a01
        rotation_rad = math.atan2(k1_sin, k1_cos)
        k1 = math.hypot(k1_cos, k1_sin)

        # Both expressions for k2 are exact; divide by the larger of |cos| and |sin| to keep the
        # quotient well conditioned near the axes.
        if abs(k1_cos) >= abs(k1_sin):
            k2 = k1 * self.a11 / k1_cos
        else:
            k2 = k1 * (tan_shear * self.a11 - self.a10) / k1_sin

        scale = (k1 + k2) / 2.0
        return AffineTransformComponents(
            scale=scale,
            asymmetry=(k1 - k2) / scale,
            rotation_rad=rotation_rad,
            shear_rad=math.atan(tan_shear),
        )

    def __call__(self, position: ProbePosition) -> ProbePosition:
        return ProbePosition(
            index=position.index,
            x_m=self.a00 * position.x_m + self.a01 * position.y_m + self.a02,
            y_m=self.a10 * position.x_m + self.a11 * position.y_m + self.a12,
            probe_photon_count=position.probe_photon_count,
        )

    def transform_coordinates(
        self, x_m: RealArrayType, y_m: RealArrayType
    ) -> tuple[RealArrayType, RealArrayType]:
        """Apply the transform to whole coordinate arrays, returning ``(x_m, y_m)``.

        One array per axis rather than a packed array, so there is no column order to
        agree on; the transform mixes the two axes, so it cannot be one call per axis.
        """
        return (
            self.a00 * x_m + self.a01 * y_m + self.a02,
            self.a10 * x_m + self.a11 * y_m + self.a12,
        )


def transform_probe_positions(
    positions: Iterable[ProbePosition],
    transform: AffineTransform,
    *,
    rng: numpy.random.Generator | None = None,
    jitter_radius_m: float = 0.0,
) -> Iterator[ProbePosition]:
    """Apply an affine transform to every position in *positions*, optionally jittering each one.

    Args:
        positions: Positions to transform.
        transform: Affine map applied to every position; the scan index and photon count ride
            through unchanged.
        rng: NumPy random generator. Jitter is applied only when one is supplied.
        jitter_radius_m: Radius, in meters, of the disk each position is displaced within. The
            displacement is uniform over that disk, so raising it widens the scatter without
            biasing the positions.
    """
    for position in positions:
        transformed = transform(position)

        if rng is not None:
            angle_rad = 2 * numpy.pi * rng.uniform()
            radius_m = jitter_radius_m * numpy.sqrt(rng.uniform())
            transformed = ProbePosition(
                index=transformed.index,
                x_m=transformed.x_m + radius_m * numpy.cos(angle_rad),
                y_m=transformed.y_m + radius_m * numpy.sin(angle_rad),
                probe_photon_count=transformed.probe_photon_count,
            )

        yield transformed


def _is_degenerate_sample(points: RealArrayType, eps: float) -> bool:
    """Return True if three points are (near-)collinear and cannot pin down a full affine.

    The test is the sine of the angle between the two edges leaving the first point, so it does
    not depend on how large the scan is. Comparing the bare cross product against an absolute
    tolerance instead would make the verdict a function of the units: an area tolerance strict
    enough for a scan tens of microns across calls every triple of a five-micron scan collinear.
    """
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    norm = math.hypot(v1[0], v1[1]) * math.hypot(v2[0], v2[1])

    if norm == 0.0:  # two of the three points coincide
        return True

    return bool(abs(cross) < eps * norm)


def _is_collinear_scan(coordinates_m: RealArrayType, eps: float) -> bool:
    """Return True if every point lies on one line, whatever the point count.

    A pair like this cannot determine a two-dimensional linear part at all, so it is worth
    separating from the ordinary case of a minimal sample that merely happened to be collinear.
    The measure is the ratio of the two singular values of the centered coordinates, which is the
    same scale-free quantity the minimal-sample screen uses.
    """
    centered = coordinates_m - coordinates_m.mean(axis=0)
    singular_values = numpy.linalg.svd(centered, compute_uv=False)

    if singular_values[0] == 0.0:  # every point coincides
        return True

    return bool(singular_values[-1] < eps * singular_values[0])


@dataclass(frozen=True)
class AffinePairFit:
    """What one (uncorrected, corrected) position pair contributed to a joint affine fit."""

    translation_x_m: float
    translation_y_m: float
    num_points: int
    num_inliers: int
    rms_residual_m: float


@dataclass(frozen=True)
class AffineFitResult:
    """Outcome of fitting a single linear part across several position pairs at once.

    ``linear_transform`` carries the shared 2x2 linear part with zero translation; each pair's own
    translation lives in the matching :class:`AffinePairFit`, where it is reported rather than
    applied, since a separate scan has a separate stage origin. Residuals are Euclidean distances
    in meters.
    """

    components: AffineTransformComponents
    linear_transform: AffineTransform
    pairs: Sequence[AffinePairFit]
    rms_residual_m: float

    def __str__(self) -> str:
        lines = [
            f'Affine fit over {len(self.pairs)} position pair(s): '
            f'scale={self.components.scale:.6f} '
            f'asymmetry={self.components.asymmetry:+.3e} '
            f'rotation={math.degrees(self.components.rotation_rad):+.4f} deg '
            f'shear={math.degrees(self.components.shear_rad):+.4f} deg; '
            f'RMS residual {self.rms_residual_m:.3e} m'
        ]

        for pair_index, pair in enumerate(self.pairs):
            lines.append(
                f'  pair {pair_index}: {pair.num_inliers}/{pair.num_points} inliers, '
                f'translation ({pair.translation_x_m:+.3e}, {pair.translation_y_m:+.3e}) m, '
                f'RMS residual {pair.rms_residual_m:.3e} m'
            )

        return '\n'.join(lines)


@dataclass(frozen=True)
class _MatchedPair:
    """One position pair reduced to index-matched coordinates and their displacement vectors."""

    coordinates_m: RealArrayType  # (N, 2) uncorrected, packed (x, y)
    displacements_m: RealArrayType  # (N, 2) corrected minus uncorrected


def _extract_indexed_coordinates(
    positions: ProbePositionSequence,
) -> tuple[IntegerArrayType, RealArrayType]:
    """Split a position sequence into its scan-index array and an (N, 2) (x, y) array."""
    return (
        positions.get_indexes(),
        numpy.column_stack((positions.get_coordinates_x_m(), positions.get_coordinates_y_m())),
    )


def _match_pair(
    uncorrected: ProbePositionSequence, corrected: ProbePositionSequence
) -> _MatchedPair:
    """Pair two position sequences by scan index and return the displacement vectors.

    Positions present on only one side are dropped, so a corrected set that lost scan points
    still contributes every point it kept.
    """
    uncorrected_indexes, uncorrected_coordinates = _extract_indexed_coordinates(uncorrected)
    corrected_indexes, corrected_coordinates = _extract_indexed_coordinates(corrected)

    _, uncorrected_rows, corrected_rows = numpy.intersect1d(
        uncorrected_indexes, corrected_indexes, assume_unique=False, return_indices=True
    )

    if uncorrected_rows.size == 0:
        raise ValueError(
            'An uncorrected/corrected position pair shares no scan indexes '
            f'({uncorrected_indexes.size} and {corrected_indexes.size} positions).'
        )

    matched_uncorrected = uncorrected_coordinates[uncorrected_rows]
    matched_corrected = corrected_coordinates[corrected_rows]
    return _MatchedPair(matched_uncorrected, matched_corrected - matched_uncorrected)


def _solve_shared_linear_offset(
    matched_pairs: Sequence[_MatchedPair], masks: Sequence[BooleanArrayType]
) -> RealArrayType:
    """Least-squares 2x2 offset ``B`` in ``displacement = B @ coordinate + translation_of_pair``.

    Each pair's own translation is eliminated by subtracting that pair's mean coordinate and mean
    displacement before stacking, so an independent stage origin per pair cannot bias ``B``.
    """
    centered_coordinates: list[RealArrayType] = []
    centered_displacements: list[RealArrayType] = []

    for pair, mask in zip(matched_pairs, masks):
        coordinates = pair.coordinates_m[mask]
        displacements = pair.displacements_m[mask]

        if coordinates.shape[0] == 0:
            continue

        centered_coordinates.append(coordinates - coordinates.mean(axis=0))
        centered_displacements.append(displacements - displacements.mean(axis=0))

    if not centered_coordinates:
        raise ValueError('No inlying probe positions remain; cannot fit an affine transform.')

    design = numpy.concatenate(centered_coordinates)
    observations = numpy.concatenate(centered_displacements)

    # Unit-RMS normalization keeps the design matrix well conditioned: scan coordinates are order
    # 1e-5 m while the displacements they explain are order 1e-8 m.
    rms_distance = numpy.sqrt(numpy.mean(numpy.sum(numpy.square(design), axis=1)))

    if rms_distance == 0.0:
        raise ValueError('All probe positions are coincident; cannot fit an affine transform.')

    normalized = design / rms_distance
    # lstsq reports the rank of the design it just factored, so the collinearity guard rides along
    # with the solve rather than paying for a second decomposition of the same matrix.
    solution, _residuals, rank, _singular_values = numpy.linalg.lstsq(
        normalized, observations, rcond=None
    )  # solution is (2, 2)

    if rank < 2:
        raise ValueError(
            'Probe positions are collinear about their per-pair centroids; '
            'the linear part of an affine transform is not determined.'
        )

    return solution.T / rms_distance


def _rms(values: RealArrayType) -> float:
    """Root mean square of a 1-D array of residual magnitudes."""
    return float(numpy.sqrt(numpy.mean(numpy.square(values))))


def _evaluate_pair_residuals(
    pair: _MatchedPair, mask: BooleanArrayType, offset: RealArrayType
) -> tuple[RealArrayType, RealArrayType]:
    """Return one pair's translation (2,) and the per-point residual magnitudes (N,)."""
    coordinates = pair.coordinates_m[mask]
    displacements = pair.displacements_m[mask]

    if coordinates.shape[0] == 0:
        translation = numpy.zeros(2)
    else:
        translation = displacements.mean(axis=0) - offset @ coordinates.mean(axis=0)

    predicted = pair.coordinates_m @ offset.T + translation
    delta = pair.displacements_m - predicted
    return translation, numpy.hypot(delta[:, 0], delta[:, 1])


def _find_pair_inliers(
    pair: _MatchedPair,
    pair_index: int,
    *,
    num_iterations: int,
    inlier_threshold_m: float,
    min_inliers: int,
    collinearity_eps: float,
    rng: numpy.random.Generator,
) -> BooleanArrayType:
    """Robustly fit one pair on its own and return the inlier mask that survives.

    The model sampled here is a full six-parameter affine, fitted to this pair alone, so one
    scan's stage origin cannot leak into another's inlier selection. Only the mask is kept; the
    linear part it implies is discarded in favor of the joint fit over every pair's inliers.
    """
    coordinates_m = pair.coordinates_m
    num_points = coordinates_m.shape[0]

    if num_points < _AFFINE_MINIMUM_POINTS:
        raise ValueError(
            f'Position pair {pair_index} has {num_points} matched point(s), but at least '
            f'{_AFFINE_MINIMUM_POINTS} are needed to fit an affine transform.'
        )

    if _is_collinear_scan(coordinates_m, collinearity_eps):
        raise ValueError(
            f'Position pair {pair_index} is one-dimensional: every point lies on a single line, '
            'so the linear part of an affine transform is not determined by it.'
        )

    # Minimal samples are screened because skimage does not screen them: fed three collinear
    # points its affine estimator returns one of the infinitely many maps that fit them and
    # reports every point an inlier, so an unscreened sample contributes a spurious candidate.
    model, inliers = ransac(
        (coordinates_m, coordinates_m + pair.displacements_m),
        skimage.transform.AffineTransform,
        min_samples=_AFFINE_MINIMUM_POINTS,
        residual_threshold=inlier_threshold_m,
        is_data_valid=lambda sample, _: not _is_degenerate_sample(sample, collinearity_eps),
        max_trials=num_iterations,
        rng=rng,
    )

    if model is None or inliers is None or numpy.count_nonzero(inliers) < min_inliers:
        raise RuntimeError(
            f'No affine transform with at least {min_inliers} inlier(s) within '
            f'{inlier_threshold_m} m was found for position pair {pair_index} in '
            f'{num_iterations} iteration(s).'
        )

    return inliers


def estimate_affine_transform(
    position_pairs: Iterable[tuple[ProbePositionSequence, ProbePositionSequence]],
    *,
    num_iterations: int = 1000,
    inlier_threshold_m: float = 1.0e-7,
    min_inliers: int = 10,
    collinearity_eps: float = 1.0e-6,
    outlier_sigma: float = 4.0,
    rng: numpy.random.Generator | None = None,
) -> AffineFitResult:
    """Fit one affine linear part jointly across several (uncorrected, corrected) position pairs.

    The fit runs on displacement vectors (corrected minus uncorrected) rather than on the corrected
    coordinates themselves. The two formulations are algebraically identical, but the displacement
    form solves for how far the transform departs from the identity, which is what carries the
    signal when the departure is five orders of magnitude smaller than the coordinates.

    Every pair shares one linear part -- the detector geometry that produced them is the same --
    while each pair keeps its own translation, since a separate scan has a separate stage origin.
    Points within a pair are matched by scan index, not by array order. Pooling pays: on three
    synthetic scans sharing one linear part, fitting each alone missed it by 1.2e-05, 2.0e-05 and
    1.7e-05 against 6.4e-06 fitting all three at once.

    The estimate is built in two stages. A random-sample consensus pass runs per pair against a
    full six-parameter affine and keeps the inliers it agrees on; the shared linear part is then
    refined by least squares over every pair's inliers together, with the per-pair translations
    profiled out by centering. Points whose refined residual exceeds ``outlier_sigma`` times a
    median-absolute-deviation estimate of the residual scale are dropped and the refinement is
    repeated. That deviation is taken about zero rather than about the median, since the residuals
    are magnitudes, and it is normalized for Gaussian data while they are Euclidean magnitudes of
    a two-dimensional error -- so ``outlier_sigma`` sets a robust multiple of the residual scale
    rather than a literal number of standard deviations.

    Args:
        position_pairs: ``(uncorrected, corrected)`` position sequences, one tuple per scan.
        num_iterations: Consensus samples drawn per pair. Raising it makes the coarse pass more
            likely to find the majority model when outliers are plentiful, at linear cost.
        inlier_threshold_m: Distance, in meters, within which the coarse pass counts a point as
            agreeing with a candidate model. Raising it admits more points and more contamination.
        min_inliers: Inliers a pair must reach for its coarse fit to be trusted. Raising it
            rejects thin agreement rather than refining on it.
        collinearity_eps: Sine of the angle below which points count as lying on one line. It
            governs both the whole-pair rejection and the screening of minimal samples; raising it
            discards more nearly-degenerate geometry.
        outlier_sigma: Residual cutoff for the refinement, as a multiple of the robust residual
            scale. Raising it keeps more points and lets a few badly refined positions pull the
            fit; lowering it trims harder. Pass ``math.inf`` to refine on the coarse inliers with
            no further trimming.
        rng: NumPy random generator for the consensus sampling. A fresh default generator is used
            when none is supplied, which makes the result irreproducible across calls.

    Returns:
        The shared components and linear part, plus the per-pair translation and residuals.

    Raises:
        ValueError: If no pairs are supplied, if a pair shares no scan indexes, if a pair has
            fewer than three matched points, or if a pair's points all lie on one line.
        RuntimeError: If the coarse pass finds no model with enough inliers for some pair.
    """
    if rng is None:
        rng = numpy.random.default_rng()

    matched_pairs = [
        _match_pair(uncorrected, corrected) for uncorrected, corrected in position_pairs
    ]

    if not matched_pairs:
        raise ValueError('At least one uncorrected/corrected position pair is required.')

    masks = [
        _find_pair_inliers(
            pair,
            pair_index,
            num_iterations=num_iterations,
            inlier_threshold_m=inlier_threshold_m,
            min_inliers=min_inliers,
            collinearity_eps=collinearity_eps,
            rng=rng,
        )
        for pair_index, pair in enumerate(matched_pairs)
    ]
    offset = _solve_shared_linear_offset(matched_pairs, masks)
    residuals = [
        _evaluate_pair_residuals(pair, mask, offset)[1] for pair, mask in zip(matched_pairs, masks)
    ]

    if math.isfinite(outlier_sigma):
        residual_scale = numpy.median(numpy.concatenate(residuals)) / _MAD_TO_SIGMA
        threshold = outlier_sigma * residual_scale

        if threshold > 0.0:
            # Intersected with the coarse masks rather than replacing them, so that a point the
            # coarse pass rejected cannot return because the refined fit happens to pass near it.
            trimmed = [mask & (residual <= threshold) for mask, residual in zip(masks, residuals)]

            if any(mask.any() for mask in trimmed):
                masks = trimmed
                offset = _solve_shared_linear_offset(matched_pairs, masks)

    pairs: list[AffinePairFit] = []
    inlier_residuals: list[RealArrayType] = []

    for pair, mask in zip(matched_pairs, masks):
        translation, residual = _evaluate_pair_residuals(pair, mask, offset)
        inlier_residuals.append(residual[mask])
        pairs.append(
            AffinePairFit(
                translation_x_m=float(translation[0]),
                translation_y_m=float(translation[1]),
                num_points=int(pair.coordinates_m.shape[0]),
                num_inliers=int(numpy.count_nonzero(mask)),
                rms_residual_m=_rms(residual[mask]) if mask.any() else 0.0,
            )
        )

    pooled = numpy.concatenate(inlier_residuals)
    linear_transform = AffineTransform(
        float(offset[0, 0]) + 1.0,
        float(offset[0, 1]),
        0.0,
        float(offset[1, 0]),
        float(offset[1, 1]) + 1.0,
        0.0,
    )
    return AffineFitResult(
        components=linear_transform.decompose(),
        linear_transform=linear_transform,
        pairs=pairs,
        rms_residual_m=_rms(pooled),
    )


def _calibrate_detector_distance(
    detector_distance_m: float,
    scale: float,
    *,
    focus_object_distance_m: float = 0.0,
) -> float:
    """Return the detector distance implied by a fitted isotropic position scale.

    In far-field geometry the object pixel is ``wavelength * distance / (width * detector pixel)``,
    so it is proportional to the detector distance -- this inverts
    :func:`ptychodus.api.propagate.compute_far_field_pixel_geometry` with respect to the distance.
    A fit that had to spread the probe positions out by ``scale`` means the assumed object pixel
    was too large by that same factor, and the distance that reproduces the correct pixel is
    ``detector_distance_m / scale``.

    Args:
        detector_distance_m: Sample-to-detector distance the positions were reconstructed with.
        scale: Isotropic scale from :meth:`AffineTransform.decompose`.
        focus_object_distance_m: Focus-to-object distance of the product. A nonzero value selects
            cone-beam geometry, where the object pixel is set by
            :func:`ptychodus.api.propagate.compute_magnification` rather than by the detector
            distance alone and this relation does not hold.

    Raises:
        ValueError: If ``scale`` is not positive and finite, if ``detector_distance_m`` is not
            finite, or if ``focus_object_distance_m`` is nonzero.
    """
    if focus_object_distance_m != 0.0:
        raise ValueError(
            'Detector-distance calibration from a position scale is defined for far-field '
            'geometry only; this product has a nonzero focus-object distance '
            f'({focus_object_distance_m} m), where the object pixel follows the magnification.'
        )

    if not math.isfinite(detector_distance_m):
        raise ValueError(f'Detector distance must be finite; got {detector_distance_m}.')

    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f'Position scale must be positive and finite; got {scale}.')

    return detector_distance_m / scale


def _target_object_geometry(
    source_geometry: ObjectGeometry,
    positions: ProbePositionSequence,
    corrected_positions: ProbePositionSequence,
    probe_geometry: ProbeGeometry,
    corrected_probe_geometry: ProbeGeometry,
    scale_free: AffineTransform,
) -> ObjectGeometry:
    """Return the grid the corrected object lives on.

    How much the canvas changes is decided by :func:`compute_object_geometry`, not here: it is
    evaluated once on the uncorrected scan and probe and once on the corrected ones, and the
    difference between the two is the change. Whatever the source object has beyond that canonical
    size -- extra padding, or simply the size a warm start read from file happens to be -- rides
    through as a signed per-axis difference, so an identity transform is an exact no-op and no
    object is silently cropped to the minimal canvas.
    """
    base_source = compute_object_geometry(positions, probe_geometry)
    base_target = compute_object_geometry(corrected_positions, corrected_probe_geometry)
    extra_width_px = source_geometry.width_px - base_source.width_px
    extra_height_px = source_geometry.height_px - base_source.height_px
    width_px = base_target.width_px + extra_width_px
    height_px = base_target.height_px + extra_height_px

    if width_px < 1 or height_px < 1:
        raise ValueError(
            f'The corrected object canvas is {width_px} x {height_px} px; the source object is '
            f'{-extra_width_px} x {-extra_height_px} px smaller than the canvas its own scan '
            'needs, and the correction shrinks it past nothing.'
        )

    center = scale_free(ProbePosition(0, source_geometry.center_x_m, source_geometry.center_y_m))
    return ObjectGeometry(
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=base_target.pixel_width_m,
        pixel_height_m=base_target.pixel_height_m,
        center_x_m=center.x_m,
        center_y_m=center.y_m,
    )


def _resample_object(
    array: ComplexArrayType,
    source_geometry: ObjectGeometry,
    target_geometry: ObjectGeometry,
    transform: AffineTransform,
    *,
    interpolation_order: int,
) -> ComplexArrayType:
    """Resample every layer of *array* from its own grid onto the target grid.

    Source and target differ in extent, pitch and center all at once, so the source index of each
    target pixel is built explicitly: take the target pixel's physical coordinate, map it back
    through the inverse of *transform*, and express the result in the source's fractional pixel
    indexes under the centered-pixel convention. Target pixels falling outside the source array
    are filled with zeros, which is what padding an object does elsewhere.
    """
    coordinates = target_geometry.get_transverse_coordinates()
    x_m = target_geometry.center_x_m + coordinates.x_m
    y_m = target_geometry.center_y_m + coordinates.y_m

    linear = numpy.array([[transform.a00, transform.a01], [transform.a10, transform.a11]])
    inverse = numpy.linalg.inv(linear)
    dx_m = x_m - transform.a02
    dy_m = y_m - transform.a12
    source_x_m = inverse[0, 0] * dx_m + inverse[0, 1] * dy_m
    source_y_m = inverse[1, 0] * dx_m + inverse[1, 1] * dy_m

    # Centered-pixel convention: the world center sits at pixel index (N - 1) / 2.
    column = (source_x_m - source_geometry.center_x_m) / source_geometry.pixel_width_m + (
        source_geometry.width_px - 1
    ) / 2
    row = (source_y_m - source_geometry.center_y_m) / source_geometry.pixel_height_m + (
        source_geometry.height_px - 1
    ) / 2

    resampled = numpy.empty(
        (array.shape[0], target_geometry.height_px, target_geometry.width_px), dtype=array.dtype
    )

    for index, layer in enumerate(array):
        real = map_coordinates(
            layer.real, (row, column), order=interpolation_order, mode='constant', cval=0.0
        )
        imaginary = map_coordinates(
            layer.imag, (row, column), order=interpolation_order, mode='constant', cval=0.0
        )
        resampled[index] = real + 1j * imaginary

    return resampled


def transform_product(
    product: Product, transform: AffineTransform, *, interpolation_order: int = 1
) -> Product:
    """Apply an affine transform to a product by splitting it between positions and geometry.

    An isotropic scale on the probe positions and the object pixel size are the same degree of
    freedom expressed two ways, so applying the whole transform to the positions *and* correcting
    the detector distance would count the scale twice. This decomposes the transform and folds
    only ``scale`` into ``detector_distance_m``, treating the scan stage as the trustworthy
    measurement and the detector distance as the miscalibrated one. Because the far-field object
    pixel is proportional to that distance, the same scale divides both pixel geometries exactly,
    with no interpolation. The remaining scale-free part -- asymmetry, rotation and shear -- moves
    the probe positions and warps the object array onto the grid they now imply.

    The object's array therefore changes shape as well as pitch, and the two axes change
    independently: a finer pixel needs more of them to cover the same scan, and asymmetry and
    rotation act differently on each axis. The probe array is reinterpreted rather than resampled,
    since the scale-free part distorts the scan frame and not the illumination optics.

    The correction is translation-free. A translation shifts the positions and the object center
    by the same amount, so it relabels the product's lab coordinates without changing a single
    sample; relocating products against each other is a cross-scan registration job.

    Resampling costs accuracy. Measured relative error of one bilinear pass against a band-limited
    reference, as a function of how far the object's content reaches toward Nyquist: 1.3e-02 at
    20%, 1.2e-01 at 40%, 4.3e-01 at 70%. Raising ``interpolation_order`` to 3 buys roughly an
    order of magnitude on smooth content at three times the cost, and buys much less on content
    that reaches Nyquist. The warp is small in absolute terms -- a corner moves 0.43 px on a 256 px
    canvas and 1.72 px on a 1024 px one at typical fitted magnitudes -- and the dominant
    component, the isotropic scale, never reaches the interpolator at all.

    Args:
        product: Product to correct.
        transform: Affine map fitted to this product's probe positions.
        interpolation_order: Spline order passed to :func:`scipy.ndimage.map_coordinates` for the
            object array. Raising it sharpens the resampled object at superlinear cost.

    Raises:
        ValueError: If the linear part is singular or orientation-reversing, if the product is not
            far-field, or if the correction shrinks the object canvas away entirely.
    """
    metadata = product.metadata
    components = transform.decompose()
    detector_distance_m = _calibrate_detector_distance(
        metadata.detector_distance_m,
        components.scale,
        focus_object_distance_m=metadata.focus_object_distance_m,
    )
    scale_free = AffineTransform.from_components(replace(components, scale=1.0))

    probes = product.probes
    probe_geometry = probes.get_geometry()
    corrected_probe_pixel_geometry = PixelGeometry(
        width_m=probe_geometry.pixel_width_m / components.scale,
        height_m=probe_geometry.pixel_height_m / components.scale,
    )
    corrected_probe_geometry = ProbeGeometry(
        width_px=probe_geometry.width_px,
        height_px=probe_geometry.height_px,
        pixel_width_m=corrected_probe_pixel_geometry.width_m,
        pixel_height_m=corrected_probe_pixel_geometry.height_m,
    )

    positions = product.probe_positions
    corrected_positions = ProbePositionSequence(
        list(transform_probe_positions(positions, scale_free))
    )

    object_ = product.object_
    source_geometry = object_.get_geometry()
    target_geometry = _target_object_geometry(
        source_geometry,
        positions,
        corrected_positions,
        probe_geometry,
        corrected_probe_geometry,
        scale_free,
    )
    logger.info(
        'Scale %.6f moves the detector distance from %.6f m to %.6f m; applying asymmetry '
        '%+.3e, rotation %+.4f deg and shear %+.4f deg to %d probe positions and resampling the '
        'object from %s onto %s.',
        components.scale,
        metadata.detector_distance_m,
        detector_distance_m,
        components.asymmetry,
        math.degrees(components.rotation_rad),
        math.degrees(components.shear_rad),
        len(positions),
        source_geometry,
        target_geometry,
    )
    return replace(
        product,
        metadata=replace(metadata, detector_distance_m=detector_distance_m),
        probe_positions=corrected_positions,
        probes=ProbeSequence(
            array=probes.get_array(),
            opr_weights=probes.get_opr_weights_or_none(),
            pixel_geometry=corrected_probe_pixel_geometry,
        ),
        object_=Object(
            array=_resample_object(
                object_.get_array(),
                source_geometry,
                target_geometry,
                scale_free,
                interpolation_order=interpolation_order,
            ),
            pixel_geometry=target_geometry.get_pixel_geometry(),
            center=target_geometry.get_center(),
            layer_spacing_m=list(object_.layer_spacing_m),
        ),
    )
