"""Tests for the dead-leaves object generator in ptychodus.api.simulate.object.

The generator claims an exact sample of the stationary dead-leaves model, so most checks
here are statistical: each sampler against the law it claims, and the generated field's
two-point statistic against the model's closed form. Equal pixel values mark the same
leaf only without anti-aliasing, so those checks use one subsample per pixel.
"""

from __future__ import annotations

import math
from typing import Any

import numpy
import pytest
from scipy import stats

from ptychodus.api.constants import PI
from ptychodus.api.object import ObjectGeometry
from ptychodus.api.simulate.object import (
    _integrate_power_law,
    _propose_leaf_hitting_points,
    _sample_truncated_power_law,
    _sample_window_leaves,
    generate_dead_leaves_object,
)
from ptychodus.api.typing import ComplexArrayType, RealArrayType

_P_VALUE_FLOOR = 1.0e-3


def _geometry(
    width_px: int = 64,
    height_px: int = 64,
    *,
    pixel_width_m: float = 1.0e-8,
    pixel_height_m: float = 1.0e-8,
) -> ObjectGeometry:
    return ObjectGeometry(
        width_px=width_px,
        height_px=height_px,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
        center_x_m=1.0e-6,
        center_y_m=-2.0e-6,
    )


def _generate(
    seed: int,
    geometry: ObjectGeometry,
    *,
    radius_lower_px: float = 1.0,
    radius_upper_px: float = 64.0,
    exponent: float = 3.0,
    amplitude_lower: float = 0.0,
    amplitude_upper: float = 1.0,
    phase_lower_rad: float = -PI,
    phase_upper_rad: float = PI,
    num_subsamples_per_axis: int = 1,
) -> ComplexArrayType:
    object_ = generate_dead_leaves_object(
        numpy.random.default_rng(seed),
        geometry,
        leaf_radius_lower_px=radius_lower_px,
        leaf_radius_upper_px=radius_upper_px,
        leaf_radius_power_law_exponent=exponent,
        leaf_amplitude_lower=amplitude_lower,
        leaf_amplitude_upper=amplitude_upper,
        leaf_phase_lower_rad=phase_lower_rad,
        leaf_phase_upper_rad=phase_upper_rad,
        num_subsamples_per_axis=num_subsamples_per_axis,
    )
    return object_.get_layer(0)


def _power_law_cdf(r: RealArrayType, lower: float, upper: float, exponent: float) -> RealArrayType:
    if math.isclose(exponent, 1.0):
        return numpy.log(r / lower) / math.log(upper / lower)

    power = 1.0 - exponent
    return (r**power - lower**power) / (upper**power - lower**power)


def _integrate_power_law_array(
    lower: float, upper: RealArrayType, exponent: float
) -> RealArrayType:
    if math.isclose(exponent, 1.0):
        return numpy.log(upper / lower)

    power = 1.0 - exponent
    return (upper**power - lower**power) / power


def _probability_same_leaf(
    distance: RealArrayType, lower: float, upper: float, exponent: float
) -> RealArrayType:
    """P(two points `distance` apart lie in one visible leaf) = E[|D ∩ D+d|] / E[|D ∪ D+d|]."""
    radius = numpy.geomspace(lower, upper, 4000)
    weight = radius**-exponent * numpy.gradient(radius)
    d = numpy.atleast_1d(distance)[:, None]
    half_angle = numpy.arccos(numpy.clip(d / (2.0 * radius), 0.0, 1.0))
    chord = 0.5 * d * numpy.sqrt(numpy.clip(4.0 * radius**2 - d**2, 0.0, None))
    intersection = numpy.where(d < 2.0 * radius, 2.0 * radius**2 * half_angle - chord, 0.0)
    union = 2.0 * PI * radius**2 - intersection
    return (intersection * weight).sum(axis=1) / (union * weight).sum(axis=1)


def _expected_same_leaf(
    lag_x_px: float, lag_y_px: float, aspect: float, lower: float, upper: float, exponent: float
) -> float:
    """`_probability_same_leaf` averaged over the jitter of two pixels' sample points."""
    jitter = numpy.random.default_rng(12345).uniform(size=(4, 100_000))
    dx = lag_x_px + jitter[1] - jitter[0]
    dy = (lag_y_px + jitter[3] - jitter[2]) * aspect
    distance = numpy.hypot(dx, dy)
    grid = numpy.linspace(distance.min(), distance.max(), 512)
    curve = _probability_same_leaf(grid, lower, upper, exponent)
    return float(numpy.interp(distance, grid, curve).mean())


def _same_leaf_z_score(fractions: list[float], expected: float) -> float:
    values = numpy.asarray(fractions)
    standard_error = values.std(ddof=1) / math.sqrt(len(values))
    return float((values.mean() - expected) / standard_error)


class TestPowerLawSampler:
    @pytest.mark.parametrize('exponent', [-1.0, 1.0, 2.0, 3.0])
    def test_matches_the_truncated_power_law(self, exponent: float) -> None:
        rng = numpy.random.default_rng(1)
        samples = _sample_truncated_power_law(rng, 2.0, 50.0, exponent, 20_000)

        assert samples.min() >= 2.0
        assert samples.max() <= 50.0
        result = stats.kstest(samples, lambda r: _power_law_cdf(r, 2.0, 50.0, exponent))
        assert result.pvalue > _P_VALUE_FLOOR

    @pytest.mark.parametrize('exponent', [-1.0, 1.0, 3.0])
    def test_integral_matches_quadrature(self, exponent: float) -> None:
        radius = numpy.linspace(2.0, 50.0, 200_001)
        quadrature = numpy.trapezoid(radius**-exponent, radius)

        assert _integrate_power_law(2.0, 50.0, exponent) == pytest.approx(quadrature, rel=1e-6)


class TestWindowLeafSampler:
    def test_radii_are_weighted_by_the_widened_window(self) -> None:
        lower, upper, exponent, width, height = 1.0, 80.0, 3.0, 40.0, 20.0
        rng = numpy.random.default_rng(2)
        _, _, radius = _sample_window_leaves(rng, lower, upper, exponent, width, height, 20_000)

        def cdf(r: RealArrayType) -> RealArrayType:
            terms = [
                (width * height, exponent),
                (2.0 * (width + height), exponent - 1.0),
                (4.0, exponent - 2.0),
            ]
            below = numpy.sum(
                [c * _integrate_power_law_array(lower, r, e) for c, e in terms], axis=0
            )
            total = sum(c * _integrate_power_law(lower, upper, e) for c, e in terms)
            return numpy.asarray(below / total)

        assert stats.kstest(radius, cdf).pvalue > _P_VALUE_FLOOR

    def test_centers_are_uniform_in_the_widened_window(self) -> None:
        width, height = 40.0, 20.0
        rng = numpy.random.default_rng(3)
        x, y, radius = _sample_window_leaves(rng, 1.0, 80.0, 3.0, width, height, 20_000)

        assert stats.kstest((x + radius) / (width + 2.0 * radius), 'uniform').pvalue > 1e-3
        assert stats.kstest((y + radius) / (height + 2.0 * radius), 'uniform').pvalue > 1e-3


class TestLeafHittingPoints:
    def test_matches_brute_force_rejection(self) -> None:
        lower, upper, exponent, width, height = 1.0, 30.0, 3.0, 20.0, 20.0
        points_x = numpy.array([2.0, 2.5, 3.0, 15.0, 10.0])
        points_y = numpy.array([2.0, 2.2, 3.5, 16.0, 5.0])
        rng = numpy.random.default_rng(4)

        direct = [
            _propose_leaf_hitting_points(rng, points_x, points_y, lower, upper, exponent)
            for _ in range(4000)
        ]
        direct_radius = numpy.array([leaf[2] for leaf in direct])
        direct_x = numpy.array([leaf[0] for leaf in direct])

        brute_radius: list[RealArrayType] = []
        brute_x: list[RealArrayType] = []

        while sum(len(radius) for radius in brute_radius) < 4000:
            x, y, radius = _sample_window_leaves(
                rng, lower, upper, exponent, width, height, 100_000
            )
            distance = numpy.hypot(x[:, None] - points_x, y[:, None] - points_y)
            hits = (distance <= radius[:, None]).any(axis=1)
            brute_radius.append(radius[hits])
            brute_x.append(x[hits])

        assert stats.ks_2samp(direct_radius, numpy.concatenate(brute_radius)).pvalue > 1e-3
        assert stats.ks_2samp(direct_x, numpy.concatenate(brute_x)).pvalue > 1e-3

    def test_claims_only_covered_points(self) -> None:
        points_x = numpy.array([0.0, 1.0, 50.0])
        points_y = numpy.array([0.0, 0.0, 50.0])
        rng = numpy.random.default_rng(5)

        for _ in range(200):
            x, y, radius, is_covered, num_proposals = _propose_leaf_hitting_points(
                rng, points_x, points_y, 1.0, 5.0, 3.0
            )
            distance = numpy.hypot(points_x - x, points_y - y)
            numpy.testing.assert_array_equal(is_covered, distance <= radius)
            assert is_covered.any()
            assert num_proposals >= 1


class TestTwoPointStatistic:
    """The fraction of pixel pairs that see one leaf, against the stationary model.

    Radii reach far beyond the window, where the statistic separates the exact sampler
    from both shortcuts it replaces: stopping a back-to-front painting at first coverage
    raises it, and drawing radii without weighting by the widened window lowers it.
    """

    lower = 2.0
    upper = 512.0
    exponent = 3.0

    @pytest.fixture(scope='class')
    def square_fields(self) -> list[ComplexArrayType]:
        geometry = _geometry(32, 32)
        return [
            _generate(seed, geometry, radius_lower_px=self.lower, radius_upper_px=self.upper)
            for seed in range(120)
        ]

    @pytest.mark.parametrize('lag_px', [1, 2, 4])
    def test_horizontal_pairs_match_the_model(
        self, square_fields: list[ComplexArrayType], lag_px: int
    ) -> None:
        fractions = [
            float(numpy.mean(array[:, :-lag_px] == array[:, lag_px:])) for array in square_fields
        ]

        expected = _expected_same_leaf(lag_px, 0, 1.0, self.lower, self.upper, self.exponent)
        assert abs(_same_leaf_z_score(fractions, expected)) < 4.0

    def test_non_square_pixels_give_circular_leaves(self) -> None:
        aspect = 2.0
        geometry = _geometry(32, 16, pixel_height_m=aspect * 1.0e-8)
        horizontal = []
        vertical = []

        for seed in range(80):
            array = _generate(
                seed, geometry, radius_lower_px=self.lower, radius_upper_px=self.upper
            )
            horizontal.append(float(numpy.mean(array[:, :-2] == array[:, 2:])))
            vertical.append(float(numpy.mean(array[:-1, :] == array[1:, :])))

        expected_horizontal = _expected_same_leaf(
            2, 0, aspect, self.lower, self.upper, self.exponent
        )
        expected_vertical = _expected_same_leaf(0, 1, aspect, self.lower, self.upper, self.exponent)
        assert abs(_same_leaf_z_score(horizontal, expected_horizontal)) < 4.0
        assert abs(_same_leaf_z_score(vertical, expected_vertical)) < 4.0


class TestMarks:
    def test_visible_leaves_carry_independent_marks(self) -> None:
        # Visibility depends only on geometry, so distinct visible values are i.i.d. marks.
        array = _generate(
            6,
            _geometry(128, 128),
            radius_lower_px=1.0,
            radius_upper_px=8.0,
            amplitude_lower=0.2,
            amplitude_upper=0.9,
            phase_lower_rad=-1.0,
            phase_upper_rad=2.0,
        )
        values = numpy.unique(array)

        assert len(values) > 500
        intensity_cdf = stats.uniform(loc=0.2**2, scale=0.9**2 - 0.2**2).cdf
        assert stats.kstest(numpy.abs(values) ** 2, intensity_cdf).pvalue > _P_VALUE_FLOOR
        phase_cdf = stats.uniform(loc=-1.0, scale=3.0).cdf
        assert stats.kstest(numpy.angle(values), phase_cdf).pvalue > _P_VALUE_FLOOR


class TestCoverage:
    @pytest.mark.parametrize(
        ('size_px', 'radius_upper_px', 'exponent', 'num_subsamples_per_axis'),
        [(64, 64.0, 3.0, 2), (256, 2000.0, 3.5, 1), (48, 10.0, 1.0, 3)],
    )
    def test_every_subsample_is_covered(
        self, size_px: int, radius_upper_px: float, exponent: float, num_subsamples_per_axis: int
    ) -> None:
        # A pixel's value is a mean of leaf values; any uncovered subsample would pull it to 0.
        spread_rad = 0.2
        array = _generate(
            7,
            _geometry(size_px, size_px),
            radius_upper_px=radius_upper_px,
            exponent=exponent,
            amplitude_lower=0.5,
            amplitude_upper=1.0,
            phase_lower_rad=1.0,
            phase_upper_rad=1.0 + spread_rad,
            num_subsamples_per_axis=num_subsamples_per_axis,
        )

        assert numpy.abs(array).min() >= 0.5 * math.cos(0.5 * spread_rad) - 1e-12

    def test_subsamples_blend_at_edges(self) -> None:
        array = _generate(8, _geometry(), num_subsamples_per_axis=4)
        num_values = len(numpy.unique(array))

        assert num_values > len(numpy.unique(_generate(8, _geometry())))


class TestInterface:
    def test_shape_dtype_and_geometry(self) -> None:
        geometry = _geometry(40, 24)
        object_ = generate_dead_leaves_object(
            numpy.random.default_rng(9),
            geometry,
            leaf_radius_lower_px=2.0,
            leaf_radius_upper_px=30.0,
            leaf_radius_power_law_exponent=3.0,
            leaf_amplitude_lower=0.0,
            leaf_amplitude_upper=1.0,
            leaf_phase_lower_rad=-PI,
            leaf_phase_upper_rad=PI,
        )
        array = object_.get_array()

        assert array.shape == (1, 24, 40)
        assert numpy.iscomplexobj(array)
        assert numpy.isfinite(array).all()
        assert object_.get_pixel_geometry() == geometry.get_pixel_geometry()
        assert object_.get_center() == geometry.get_center()

    def test_same_seed_reproduces_and_different_seeds_differ(self) -> None:
        geometry = _geometry(32, 32)

        numpy.testing.assert_array_equal(_generate(10, geometry), _generate(10, geometry))
        assert not numpy.array_equal(_generate(10, geometry), _generate(11, geometry))

    @pytest.mark.parametrize(
        'overrides',
        [
            {'radius_lower_px': 0.0},
            {'radius_lower_px': -1.0},
            {'radius_lower_px': 64.0, 'radius_upper_px': 64.0},
            {'amplitude_lower': 1.0, 'amplitude_upper': 0.5},
            {'phase_lower_rad': 1.0, 'phase_upper_rad': 1.0},
            {'num_subsamples_per_axis': 0},
        ],
    )
    def test_rejects_invalid_parameters(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            _generate(0, _geometry(16, 16), **overrides)
