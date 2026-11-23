"""Spectral autofocus on the simulated parallax scans."""

from __future__ import annotations

from typing import Any

import pytest

from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.defocus import AutofocusEstimate, estimate_defocus_by_autofocus

from parallax_scan_fixtures import (
    DEPTH_OF_FOCUS_M,
    METADATA,
    OBJECT_GEOMETRY,
    PROBE_GEOMETRY,
    WAVELENGTH_M,
    make_positions,
    simulate_scan,
)

# 2.3 depths of focus: far enough from focus for the residual to place it, inside the
# default near-focus guard of 3, so the guard can be tested on the same scan.
TRUE_DEFOCUS_M = 3.0e-3


def _autofocus(true_defocus_m: float, initial_defocus_m: float, **kwargs: Any) -> AutofocusEstimate:
    return estimate_defocus_by_autofocus(
        simulate_scan(true_defocus_m),
        ProbePositionSequence(make_positions()),
        OBJECT_GEOMETRY,
        PROBE_GEOMETRY,
        photon_wavelength_m=WAVELENGTH_M,
        detector_distance_m=METADATA.detector_distance_m,
        initial_defocus_m=initial_defocus_m,
        depth_of_focus_m=DEPTH_OF_FOCUS_M,
        **kwargs,
    )


@pytest.mark.parametrize('true_defocus_m', [TRUE_DEFOCUS_M, -TRUE_DEFOCUS_M])
@pytest.mark.parametrize('factor', [0.6, 1.5])
def test_recovers_the_defocus_from_a_wrong_initial_one(
    true_defocus_m: float, factor: float
) -> None:
    """The residual's minimum, not the start.

    Measured from 1.8 and 4.5 mm: 2.966 and 2.983 mm, and -2.903 mm from either start on
    the other side of focus, where the weak phase texture images with the opposite
    contrast and the residual's minimum sits 3 % nearer focus.
    """
    result = _autofocus(true_defocus_m, factor * true_defocus_m, min_defocus_dof=1.0)

    assert not result.kept_initial
    assert not result.on_edge
    assert result.defocus_distance_m == pytest.approx(true_defocus_m, rel=0.05)
    assert result.num_patterns == len(make_positions())
    # Seven candidates and the two that refine the best of them.
    assert len(result.candidates_m) == len(result.residuals) == 9


def test_the_pattern_cap_keeps_a_central_block() -> None:
    """A cap of 200 of the 361 patterns keeps the central 13 x 13 block: 169 patterns, and
    2.982 mm."""
    result = _autofocus(TRUE_DEFOCUS_M, 0.6 * TRUE_DEFOCUS_M, min_defocus_dof=1.0, max_patterns=200)

    assert 100 < result.num_patterns <= 200
    assert result.defocus_distance_m == pytest.approx(TRUE_DEFOCUS_M, rel=0.05)


def test_an_initial_defocus_within_the_depth_of_focus_is_kept_unevaluated() -> None:
    initial_m = 0.5 * DEPTH_OF_FOCUS_M

    result = _autofocus(TRUE_DEFOCUS_M, initial_m)

    assert result.kept_initial
    assert result.defocus_distance_m == initial_m
    assert result.candidates_m == ()


def test_a_minimum_near_focus_keeps_the_initial_defocus() -> None:
    """3 mm lies within the default three depths of focus, where the residual is too flat."""
    initial_m = 0.6 * TRUE_DEFOCUS_M

    result = _autofocus(TRUE_DEFOCUS_M, initial_m)

    assert result.kept_initial
    assert result.defocus_distance_m == initial_m
    assert len(result.residuals) == 9


@pytest.mark.parametrize(
    'kwargs, match',
    [
        ({'detector_distance_m': 0.0}, 'detector distance'),
        ({'depth_of_focus_m': 0.0}, 'depth of focus'),
        ({'num_candidates': 2}, 'candidates'),
        ({'max_patterns': 0}, 'one pattern'),
    ],
)
def test_invalid_inputs_are_refused(kwargs: dict[str, float], match: str) -> None:
    arguments: dict[str, Any] = {
        'photon_wavelength_m': WAVELENGTH_M,
        'detector_distance_m': METADATA.detector_distance_m,
        'initial_defocus_m': TRUE_DEFOCUS_M,
        'depth_of_focus_m': DEPTH_OF_FOCUS_M,
        **kwargs,
    }

    with pytest.raises(ValueError, match=match):
        estimate_defocus_by_autofocus(
            simulate_scan(TRUE_DEFOCUS_M),
            ProbePositionSequence(make_positions()),
            OBJECT_GEOMETRY,
            PROBE_GEOMETRY,
            **arguments,
        )
