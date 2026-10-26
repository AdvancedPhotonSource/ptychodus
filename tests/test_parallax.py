"""Unit tests for the parallax between bright-field images: the defocus, and the orientation.

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
  - the eight detector orientations: each map is what its prep steps do to a pixel, and
    a half turn is negation
  - the matrix fit names the orientation of patterns flipped or transposed after
    simulation -- the positive-defocus reading's steps restore them exactly -- tells a
    reflection apart, and reads a rotated scan as a residual rotation
"""

from collections.abc import Callable
from functools import cache
import math

import numpy
import pytest

from parallax_scan_fixtures import (
    DEPTH_OF_FOCUS_M,
    OBJECT_GEOMETRY,
    PIXEL_SIZE_M,
    PROBE_GEOMETRY,
    WAVELENGTH_M,
    accumulate_scan,
    make_positions,
    simulate_scan,
)
from ptychodus.api.accumulate import DiffractionMoments, compute_diffraction_moments
from ptychodus.api.parallax import (
    DetectorOrientation,
    ParallaxEstimate,
    ParallaxTransform,
    estimate_defocus_from_parallax,
    estimate_parallax_transform,
)
from ptychodus.api.preprocess.diffraction import BinningStep, DiffractionPrepPipeline
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


# ---------------------------------------------------------------------------
# Detector orientation
# ---------------------------------------------------------------------------


def test_there_are_eight_orientations_the_identity_first() -> None:
    orientations = DetectorOrientation.create_all()

    assert len(orientations) == 8
    assert orientations[0] == DetectorOrientation()
    numpy.testing.assert_array_equal(orientations[0].matrix, numpy.eye(2))
    assert len({tuple(o.matrix.ravel()) for o in orientations}) == 8


@pytest.mark.parametrize('orientation', DetectorOrientation.create_all(), ids=repr)
def test_an_orientations_map_is_what_its_steps_do_to_a_pixel(
    orientation: DetectorOrientation,
) -> None:
    """A lone pixel moves, about the frame midpoint, exactly as the map says."""
    frame = numpy.zeros((1, 9, 9))
    x, y = 3, -2
    frame[0, 4 + y, 4 + x] = 1.0

    moved = DiffractionPrepPipeline(steps=orientation.prep_steps()).apply_to_patterns(frame)

    row, column = numpy.argwhere(moved[0])[0]
    expected_x, expected_y = orientation.matrix @ numpy.array([x, y])
    assert (column - 4, row - 4) == (expected_x, expected_y)


@pytest.mark.parametrize('orientation', DetectorOrientation.create_all(), ids=repr)
def test_a_half_turn_is_negation_and_an_involution(orientation: DetectorOrientation) -> None:
    turned = orientation.rotated_by_half_turn()

    numpy.testing.assert_array_equal(turned.matrix, -orientation.matrix)
    assert turned.rotated_by_half_turn() == orientation


_RAW_FRAMES: dict[str, Callable[[numpy.ndarray], numpy.ndarray]] = {
    'identity': lambda frames: frames,
    'hflip': lambda frames: numpy.flip(frames, -1),
    'vflip': lambda frames: numpy.flip(frames, -2),
    'transpose': lambda frames: numpy.swapaxes(frames, -1, -2),
    'hflip_then_transpose': lambda frames: numpy.swapaxes(numpy.flip(frames, -1), -1, -2),
    'half_turn': lambda frames: numpy.flip(frames, (-2, -1)),
}
_REFLECTIONS = {'hflip', 'vflip', 'transpose'}


@cache
def _transform(raw_frames: str, rotation_deg: float = 0.0) -> ParallaxTransform:
    """The matrix fit for the +2 mm scan, its patterns reoriented and its positions turned.

    Turning the recorded positions from x towards y while the patterns stay as simulated
    is a scan rotated relative to the detector.
    """
    data = simulate_scan(2.0e-3)
    patterns = _RAW_FRAMES[raw_frames](numpy.array(data.get_patterns()))
    moments = compute_diffraction_moments(
        numpy.array(data.get_indexes()),
        patterns,
        bad_pixels=numpy.zeros(patterns.shape[-2:], dtype=bool),
        binning=BinningStep(bin_size_x=2, bin_size_y=2),
    )
    assert moments is not None
    cos, sin = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    positions = [
        ProbePosition(p.index, cos * p.x_m - sin * p.y_m, sin * p.x_m + cos * p.y_m)
        for p in make_positions()
    ]
    transform = estimate_parallax_transform(
        moments,
        ProbePositionSequence(positions),
        OBJECT_GEOMETRY,
        PROBE_GEOMETRY,
        photon_wavelength_m=WAVELENGTH_M,
    )
    assert transform is not None
    return transform


@pytest.mark.parametrize('raw_frames', list(_RAW_FRAMES))
def test_the_positive_defocus_reading_restores_reoriented_patterns(raw_frames: str) -> None:
    transform = _transform(raw_frames)
    first, second = transform.orientation_candidates()
    standard = numpy.array(simulate_scan(2.0e-3).get_patterns())
    raw = _RAW_FRAMES[raw_frames](standard)

    restored = DiffractionPrepPipeline(steps=first.orientation.prep_steps()).apply_to_patterns(raw)

    numpy.testing.assert_array_equal(restored, standard)
    # Measured at 2.009 mm whatever the orientation.
    assert first.defocus_distance_m == pytest.approx(2.0e-3, rel=0.06)
    assert second.orientation == first.orientation.rotated_by_half_turn()
    assert second.defocus_distance_m == -first.defocus_distance_m
    assert transform.is_reflection == (raw_frames in _REFLECTIONS)
    assert transform.fresnel_number > 1.0


def test_an_aligned_scan_leaves_no_rotation_asymmetry_or_shear() -> None:
    components = _transform('identity').components()

    # Measured: 0.26 deg, 0.030 and 0.41 deg.
    assert abs(math.degrees(components.rotation_rad)) < 1.0
    assert abs(components.asymmetry) < 0.06
    assert abs(math.degrees(components.shear_rad)) < 1.5


def test_a_rotated_scan_reads_as_a_residual_rotation() -> None:
    transform = _transform('identity', rotation_deg=10.0)

    assert transform.orientation_candidates()[0].orientation == DetectorOrientation()
    # Measured -10.19 deg: AffineTransformComponents' rotation is the standard one negated.
    assert math.degrees(transform.components().rotation_rad) == pytest.approx(-10.0, abs=1.0)


def test_a_matrix_fit_needs_four_bins() -> None:
    with pytest.raises(ValueError, match='four bins'):
        estimate_parallax_transform(
            accumulate_scan(2.0e-3),
            ProbePositionSequence(make_positions()),
            OBJECT_GEOMETRY,
            PROBE_GEOMETRY,
            photon_wavelength_m=WAVELENGTH_M,
            minimum_bins=3,
        )
