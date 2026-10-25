"""Unit tests for estimate_defocus_from_parallax: defocus from shifted bright-field images.

Every scan is simulated by the far-field forward model, with a probe whose far field is a
uniform disk, propagated from its focus by the defocus under test. The sign of the
estimate is therefore pinned against that model rather than derived.

Behaviors verified:
  - a defocus beyond the depth of focus is recovered with its sign, pairing patterns
    with positions by index rather than by order
  - the depth of focus is the disk's ``lambda / NA**2 = 1 / (lambda * q_max**2)``, and
    the Fresnel number is the defocus in units of it
  - a probe at focus and a featureless sample are not resolved, and the depth-of-focus
    threshold is what rejects the probe at focus
  - too few bright-field bins give None; moments without binned patterns, or frames
    that do not match the probe, are rejected
"""

import pytest

from parallax_scan_fixtures import (
    DEPTH_OF_FOCUS_M,
    OBJECT_GEOMETRY,
    PIXEL_SIZE_M,
    PROBE_GEOMETRY,
    WAVELENGTH_M,
    accumulate_scan,
    make_positions,
)
from ptychodus.api.accumulate import DiffractionMoments
from ptychodus.api.parallax import ParallaxEstimate, estimate_defocus_from_parallax
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence


def _estimate(
    moments: DiffractionMoments, positions: list[ProbePosition] | None = None
) -> ParallaxEstimate | None:
    return estimate_defocus_from_parallax(
        moments,
        ProbePositionSequence(make_positions() if positions is None else positions),
        OBJECT_GEOMETRY,
        PROBE_GEOMETRY,
        photon_wavelength_m=WAVELENGTH_M,
    )


@pytest.mark.parametrize('defocus_m', [2.0e-3, -2.0e-3])
def test_a_defocus_is_recovered_with_its_sign(defocus_m: float) -> None:
    # Reversed, so pairing by order rather than by index would scramble the images.
    estimate = _estimate(accumulate_scan(defocus_m), make_positions()[::-1])

    assert estimate is not None
    # Measured within 4% at +-2 mm on this scan.
    assert estimate.defocus_distance_m == pytest.approx(defocus_m, rel=0.06)
    assert estimate.is_resolved()


def test_the_depth_of_focus_is_that_of_the_disk() -> None:
    estimate = _estimate(accumulate_scan(2.0e-3))

    assert estimate is not None
    # Second moments of the noisy pattern after its noise floor: measured within 4%.
    assert estimate.depth_of_focus_m == pytest.approx(DEPTH_OF_FOCUS_M, rel=0.08)
    assert estimate.fresnel_number == pytest.approx(
        abs(estimate.defocus_distance_m) / estimate.depth_of_focus_m
    )


def test_a_probe_at_focus_is_not_resolved() -> None:
    estimate = _estimate(accumulate_scan(0.0))

    # Its images differ by phase contrast, not by a shift, so the fit finds a slope
    # with little scatter -- but no larger than the depth of focus.
    assert estimate is not None
    assert estimate.fresnel_number < 1.0
    assert estimate.significance > 3.0
    assert not estimate.is_resolved()
    assert estimate.is_resolved(minimum_defocus_dof=0.5 * estimate.fresnel_number)


def test_a_featureless_sample_is_not_resolved() -> None:
    estimate = _estimate(accumulate_scan(2.0e-3, featureless=True))

    # Its images are independent photon noise, so the slope is too: measured at about
    # two standard errors on this scan, against three needed.
    assert estimate is not None
    assert estimate.significance < 3.0
    assert not estimate.is_resolved()


def test_too_few_bright_field_bins_give_none() -> None:
    # Bins of 8 pixels leave a disk of radius 8 with no bin away from its edge.
    assert _estimate(accumulate_scan(2.0e-3, bin_size_px=8)) is None


def test_moments_without_binned_patterns_are_rejected() -> None:
    moments = accumulate_scan(2.0e-3)
    unbinned = DiffractionMoments(
        num_patterns=moments.num_patterns,
        minimum_frame=moments.minimum_frame,
        mean_frame=moments.mean_frame,
        maximum_frame=moments.maximum_frame,
        bad_pixels=moments.bad_pixels,
        indexes=moments.indexes,
        total_counts=moments.total_counts,
        center_of_mass_x_px=moments.center_of_mass_x_px,
        center_of_mass_y_px=moments.center_of_mass_y_px,
    )

    with pytest.raises(ValueError, match='binning step'):
        _estimate(unbinned)


def test_frames_that_do_not_match_the_probe_are_rejected() -> None:
    probe_geometry = ProbeGeometry(
        width_px=32, height_px=32, pixel_width_m=PIXEL_SIZE_M, pixel_height_m=PIXEL_SIZE_M
    )

    with pytest.raises(ValueError, match='do not match'):
        estimate_defocus_from_parallax(
            accumulate_scan(2.0e-3),
            ProbePositionSequence(make_positions()),
            OBJECT_GEOMETRY,
            probe_geometry,
            photon_wavelength_m=WAVELENGTH_M,
        )
