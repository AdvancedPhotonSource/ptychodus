"""Tests for the multislice layer transforms in ptychodus.api.object.

Currently covers:
  - compute_uniform_layer_spacing_m: the slab convention and its degenerate cases.
  - select_object_layers: index selection and the depth grid it derives.
  - homogenize_object_layers: redistributing the layer product, past +/-pi.
  - scale_object_phase: phase scaling and when it needs unwrapping.
  - resize_object_layers: growing and shrinking without interpolation.
  - resample_object_layers: interpolating onto a new depth grid.
"""

from __future__ import annotations

import numpy
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import (
    LayerFillMode,
    Object,
    ObjectCenter,
    compute_uniform_layer_spacing_m,
    homogenize_object_layers,
    resample_object_layers,
    resize_object_layers,
    scale_object_phase,
    select_object_layers,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EXTENT_PX = 8
PIXEL_SIZE_M = 1.0e-9
THICKNESS_M = 2.0e-5


def _pixel_geometry() -> PixelGeometry:
    return PixelGeometry(width_m=PIXEL_SIZE_M, height_m=PIXEL_SIZE_M)


def _make_object(array: numpy.ndarray, layer_spacing_m: list[float] | None = None) -> Object:
    num_layers = array.shape[0]

    return Object(
        array=array.astype(complex),
        pixel_geometry=_pixel_geometry(),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=compute_uniform_layer_spacing_m(THICKNESS_M, num_layers)
        if layer_spacing_m is None
        else layer_spacing_m,
    )


def _make_layered_object(num_layers: int, *, seed: int = 1) -> Object:
    """A depth-varying, spatially varying stack whose per-layer phase stays below pi."""
    rng = numpy.random.default_rng(seed)
    profile = numpy.linspace(0.7, 1.4, num_layers)[:, None, None]
    texture = rng.uniform(0.5, 1.1, (EXTENT_PX, EXTENT_PX))[None]

    return _make_object((0.97**profile) * numpy.exp(1j * profile * texture))


def _make_constant_phase_object(phases_rad: list[float]) -> Object:
    """A stack of uniform layers, for exact analytic assertions."""
    array = numpy.stack(
        [numpy.full((EXTENT_PX, EXTENT_PX), numpy.exp(1j * phase)) for phase in phases_rad]
    )

    return _make_object(array)


# ---------------------------------------------------------------------------
# compute_uniform_layer_spacing_m
# ---------------------------------------------------------------------------


def test_uniform_layer_spacing_single_layer_is_empty() -> None:
    assert compute_uniform_layer_spacing_m(THICKNESS_M, 1) == []


@pytest.mark.parametrize('num_layers', [2, 3, 5, 17])
def test_uniform_layer_spacing_sums_to_the_requested_thickness(num_layers: int) -> None:
    """The layers are screens spanning the slab, so the gaps must add up to the whole of it."""
    spacing_m = compute_uniform_layer_spacing_m(THICKNESS_M, num_layers)

    assert len(spacing_m) == num_layers - 1
    numpy.testing.assert_allclose(sum(spacing_m), THICKNESS_M, atol=1.0e-18)
    numpy.testing.assert_allclose(spacing_m, THICKNESS_M / (num_layers - 1), atol=1.0e-18)


@pytest.mark.parametrize('num_layers', [1, 2, 5])
def test_uniform_layer_spacing_agrees_with_get_total_thickness_m(num_layers: int) -> None:
    """Pins the two against each other so the convention cannot drift apart again."""
    obj = _make_layered_object(num_layers)
    expected_m = THICKNESS_M if num_layers > 1 else 0.0

    numpy.testing.assert_allclose(obj.get_total_thickness_m(), expected_m, atol=1.0e-18)


def test_uniform_layer_spacing_zero_thickness_keeps_the_layer_count() -> None:
    """A zero-thickness multislice stack used to raise from the Object constructor."""
    spacing_m = compute_uniform_layer_spacing_m(0.0, 5)

    assert spacing_m == [0.0] * 4
    assert _make_object(numpy.ones((5, EXTENT_PX, EXTENT_PX)), spacing_m).num_layers == 5


@pytest.mark.parametrize('num_layers', [0, -3])
def test_uniform_layer_spacing_rejects_a_nonpositive_layer_count(num_layers: int) -> None:
    with pytest.raises(ValueError):
        compute_uniform_layer_spacing_m(THICKNESS_M, num_layers)


# ---------------------------------------------------------------------------
# select_object_layers
# ---------------------------------------------------------------------------


def test_select_object_layers_keeps_the_requested_layers() -> None:
    obj = _make_layered_object(6)
    selected = select_object_layers(obj, [0, 2, 5])

    assert selected.num_layers == 3
    numpy.testing.assert_allclose(selected.get_array(), obj.get_array()[[0, 2, 5]], atol=1.0e-15)


def test_select_object_layers_derives_spacing_from_the_depth_grid() -> None:
    """A non-contiguous selection must record the true distance between the kept layers."""
    obj = _make_object(numpy.ones((5, EXTENT_PX, EXTENT_PX)), [1.0, 2.0, 3.0, 4.0])
    selected = select_object_layers(obj, [0, 2, 4])

    numpy.testing.assert_allclose(list(selected.layer_spacing_m), [3.0, 7.0], atol=1.0e-15)


def test_select_object_layers_single_index_has_no_spacing() -> None:
    selected = select_object_layers(_make_layered_object(4), [2])

    assert selected.num_layers == 1
    assert list(selected.layer_spacing_m) == []


def test_select_object_layers_preserves_pixel_geometry_and_center() -> None:
    selected = select_object_layers(_make_layered_object(4), [1, 3])

    assert selected.get_pixel_geometry().width_m == PIXEL_SIZE_M
    assert selected.get_center().x_m == 0.0


def test_select_object_layers_rejects_an_out_of_range_index() -> None:
    with pytest.raises(IndexError):
        select_object_layers(_make_layered_object(3), [0, 7])


def test_select_object_layers_drops_out_of_range_indexes_when_asked() -> None:
    selected = select_object_layers(_make_layered_object(3), [0, 7], drop_out_of_range=True)

    assert selected.num_layers == 1


def test_select_object_layers_rejects_a_wholly_out_of_range_selection() -> None:
    """An object must keep at least one layer, so dropping everything is still an error."""
    with pytest.raises(ValueError):
        select_object_layers(_make_layered_object(3), [9], drop_out_of_range=True)


def test_select_object_layers_rejects_an_empty_selection() -> None:
    with pytest.raises(ValueError):
        select_object_layers(_make_layered_object(3), [])


@pytest.mark.parametrize('indexes', [[2, 0], [1, 1]])
def test_select_object_layers_rejects_a_non_increasing_selection(indexes: list[int]) -> None:
    """Out-of-order or repeated indexes would make the derived depth grid meaningless."""
    with pytest.raises(ValueError):
        select_object_layers(_make_layered_object(4), indexes)


# ---------------------------------------------------------------------------
# homogenize_object_layers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('num_layers', [1, 2, 4, 9])
def test_homogenize_object_layers_preserves_the_layer_product(num_layers: int) -> None:
    obj = _make_layered_object(4)
    homogenized = homogenize_object_layers(
        obj,
        num_layers=num_layers,
        layer_spacing_m=compute_uniform_layer_spacing_m(THICKNESS_M, num_layers),
    )

    assert homogenized.num_layers == num_layers
    numpy.testing.assert_allclose(
        homogenized.get_layers_flattened(), obj.get_layers_flattened(), atol=1.0e-12
    )


def test_homogenize_object_layers_to_one_layer_is_the_flattened_object() -> None:
    obj = _make_layered_object(5)
    homogenized = homogenize_object_layers(obj, num_layers=1)

    numpy.testing.assert_allclose(
        homogenized.get_array()[0], obj.get_layers_flattened(), atol=1.0e-12
    )


def test_homogenize_object_layers_makes_every_layer_identical() -> None:
    homogenized = homogenize_object_layers(_make_layered_object(4))
    array = homogenized.get_array()

    for layer in array[1:]:
        numpy.testing.assert_allclose(layer, array[0], atol=1.0e-15)


def test_homogenize_object_layers_survives_a_total_phase_beyond_pi() -> None:
    """Regression pin for the wrapping defect.

    Three layers of 2.0 rad total 6.0 rad, which is outside ``(-pi, pi]``. Dividing the
    phase of the layer product -- the old behaviour -- would wrap that to ``6 - 2*pi``
    and give each layer -0.094 rad instead of 2.0.
    """
    obj = _make_constant_phase_object([2.0, 2.0, 2.0])
    homogenized = homogenize_object_layers(obj, num_layers=3)

    numpy.testing.assert_allclose(numpy.angle(homogenized.get_array()), 2.0, atol=1.0e-12)


def _make_wrapped_phase_object() -> Object:
    """One layer whose phase ramps past 2 pi, so numpy.angle wraps it into (-pi, pi].

    Every wrap draws one contour of adjacent-pixel jumps larger than pi, so a ramp of
    several turns leaves a fraction of jumps well above the default threshold.
    """
    ramp = numpy.linspace(0.0, 12.0 * numpy.pi, EXTENT_PX * EXTENT_PX)
    layer = numpy.exp(1j * ramp).reshape(1, EXTENT_PX, EXTENT_PX)

    return _make_object(layer)


def test_homogenize_object_layers_warns_about_a_wrapped_layer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The principal logarithm of a wrapped layer is not its optical path, and nothing
    # downstream reveals that, so the warning is the only signal the caller gets.
    with caplog.at_level('WARNING', logger='ptychodus.api.object'):
        homogenize_object_layers(_make_wrapped_phase_object())

    assert 'looks phase-wrapped' in caplog.text


def test_a_raised_jump_fraction_accepts_a_finely_textured_layer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # What the knob is for: an object whose own structure jumps by more than pi between
    # neighboring pixels is not wrapped, and should not be reported as though it were.
    with caplog.at_level('WARNING', logger='ptychodus.api.object'):
        homogenize_object_layers(_make_wrapped_phase_object(), wrapped_phase_jump_fraction=1.0)

    assert caplog.text == ''


def test_unwrapping_silences_the_warning_whatever_the_fraction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The fraction gates a report about unwrapped phase, so it has nothing to say once
    # the phase has been unwrapped -- even at a threshold that would flag anything.
    with caplog.at_level('WARNING', logger='ptychodus.api.object'):
        homogenize_object_layers(
            _make_wrapped_phase_object(),
            unwrap_phase_rad=True,
            wrapped_phase_jump_fraction=0.0,
        )

    assert caplog.text == ''


def test_resample_object_layers_forwards_the_jump_fraction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    obj = _make_object(
        numpy.repeat(_make_wrapped_phase_object().get_array(), 2, axis=0),
    )

    with caplog.at_level('WARNING', logger='ptychodus.api.object'):
        resample_object_layers(obj, [THICKNESS_M / 2.0] * 2, wrapped_phase_jump_fraction=1.0)

    assert caplog.text == ''


def test_resize_object_layers_forwards_the_jump_fraction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Only the geometric-mean fill takes a logarithm, so it is the one path here that
    # can report a wrap at all.
    obj = _make_wrapped_phase_object()

    with caplog.at_level('WARNING', logger='ptychodus.api.object'):
        resize_object_layers(
            obj, 3, fill_mode=LayerFillMode.GEOMETRIC_MEAN, wrapped_phase_jump_fraction=1.0
        )

    assert caplog.text == ''


def test_homogenize_object_layers_single_layer_input_is_unchanged() -> None:
    obj = _make_layered_object(1)

    numpy.testing.assert_allclose(
        homogenize_object_layers(obj).get_array(), obj.get_array(), atol=1.0e-12
    )


def test_homogenize_object_layers_requires_spacing_when_the_count_changes() -> None:
    with pytest.raises(ValueError):
        homogenize_object_layers(_make_layered_object(3), num_layers=5)


def test_homogenize_object_layers_rejects_a_wrong_length_spacing() -> None:
    with pytest.raises(ValueError):
        homogenize_object_layers(_make_layered_object(3), num_layers=4, layer_spacing_m=[1.0])


# ---------------------------------------------------------------------------
# scale_object_phase
# ---------------------------------------------------------------------------


def test_scale_object_phase_unit_scaling_is_the_same_object() -> None:
    obj = _make_layered_object(3)

    assert scale_object_phase(obj, 1.0) is obj


def test_scale_object_phase_leaves_the_amplitude_alone() -> None:
    obj = _make_layered_object(3)
    scaled = scale_object_phase(obj, 2.0)

    numpy.testing.assert_allclose(
        numpy.abs(scaled.get_array()), numpy.abs(obj.get_array()), atol=1.0e-15
    )


def test_scale_object_phase_doubles_a_small_phase() -> None:
    obj = _make_constant_phase_object([0.3])
    scaled = scale_object_phase(obj, 2.0)

    numpy.testing.assert_allclose(numpy.angle(scaled.get_array()), 0.6, atol=1.0e-12)


@pytest.mark.parametrize('scaling', [2.0, 3.0, -2.0])
def test_scale_object_phase_integer_scaling_needs_no_unwrapping(scaling: float) -> None:
    """An integer factor turns the stored phase's 2-pi ambiguity into a whole number of
    turns, which cancels -- so the wrapped and unwrapped answers agree exactly."""
    true_phase_rad = 4.5
    obj = _make_object(numpy.full((1, EXTENT_PX, EXTENT_PX), 0.9 * numpy.exp(1j * true_phase_rad)))
    scaled = scale_object_phase(obj, scaling, unwrap_phase_rad=False)

    numpy.testing.assert_allclose(
        scaled.get_array(),
        0.9 * numpy.exp(1j * true_phase_rad * scaling),
        atol=1.0e-12,
    )


def test_scale_object_phase_fractional_scaling_differs_when_wrapped() -> None:
    """The counterpart: a fractional factor does not cancel, which is why the default
    unwraps for one."""
    true_phase_rad = 4.5
    obj = _make_object(numpy.full((1, EXTENT_PX, EXTENT_PX), 0.9 * numpy.exp(1j * true_phase_rad)))
    scaled = scale_object_phase(obj, 2.5, unwrap_phase_rad=False)

    assert not numpy.allclose(
        scaled.get_array(), 0.9 * numpy.exp(1j * true_phase_rad * 2.5), atol=1.0e-3
    )


def test_scale_object_phase_preserves_layer_spacing() -> None:
    obj = _make_layered_object(4)
    scaled = scale_object_phase(obj, 2.0)

    numpy.testing.assert_allclose(
        list(scaled.layer_spacing_m), list(obj.layer_spacing_m), atol=1.0e-18
    )


# ---------------------------------------------------------------------------
# resize_object_layers
# ---------------------------------------------------------------------------


def test_resize_object_layers_to_the_same_count_is_the_same_object() -> None:
    obj = _make_layered_object(4)

    assert resize_object_layers(obj, 4) is obj


def test_resize_object_layers_center_crops_when_shrinking() -> None:
    obj = _make_layered_object(5)
    resized = resize_object_layers(obj, 3)

    numpy.testing.assert_allclose(resized.get_array(), obj.get_array()[1:4], atol=1.0e-15)


def test_resize_object_layers_shrink_slices_the_spacing_to_match() -> None:
    obj = _make_object(numpy.ones((5, EXTENT_PX, EXTENT_PX)), [1.0, 2.0, 3.0, 4.0])
    resized = resize_object_layers(obj, 3)

    numpy.testing.assert_allclose(list(resized.layer_spacing_m), [2.0, 3.0], atol=1.0e-15)


def test_resize_object_layers_to_one_layer_takes_the_product() -> None:
    obj = _make_layered_object(4)
    resized = resize_object_layers(obj, 1)

    numpy.testing.assert_allclose(resized.get_array()[0], obj.get_layers_flattened(), atol=1.0e-15)
    assert list(resized.layer_spacing_m) == []


def test_resize_object_layers_grows_alternately_back_then_front() -> None:
    """Insertion order is observable, so pin it: first filler goes after the stack."""
    obj = _make_constant_phase_object([0.1, 0.2])
    resized = resize_object_layers(obj, 4, fill_mode=LayerFillMode.VACUUM)
    phases_rad = numpy.angle(resized.get_array()[:, 0, 0])

    numpy.testing.assert_allclose(phases_rad, [0.0, 0.1, 0.2, 0.0], atol=1.0e-12)


def test_resize_object_layers_vacuum_fill_preserves_the_layer_product() -> None:
    """The property that makes VACUUM the right default: growing changes no physics."""
    obj = _make_layered_object(4)
    resized = resize_object_layers(obj, 9, fill_mode=LayerFillMode.VACUUM)

    numpy.testing.assert_allclose(
        resized.get_layers_flattened(), obj.get_layers_flattened(), atol=1.0e-15
    )


def test_resize_object_layers_edge_fill_repeats_the_exit_layer() -> None:
    obj = _make_constant_phase_object([0.1, 0.2])
    resized = resize_object_layers(obj, 3, fill_mode=LayerFillMode.EDGE)

    numpy.testing.assert_allclose(resized.get_array()[-1], obj.get_array()[-1], atol=1.0e-15)


def test_resize_object_layers_geometric_mean_fill_does_not_cancel() -> None:
    """An arithmetic mean of these two layers would give |0.17|, turning transparent
    material into an absorbing slab; the geometric mean keeps unit transmission."""
    obj = _make_constant_phase_object([1.4, -1.4])
    resized = resize_object_layers(obj, 3, fill_mode=LayerFillMode.GEOMETRIC_MEAN)

    numpy.testing.assert_allclose(numpy.abs(resized.get_array()[-1]), 1.0, atol=1.0e-12)


def test_resize_object_layers_grow_repeats_the_adjacent_gap() -> None:
    obj = _make_object(numpy.ones((2, EXTENT_PX, EXTENT_PX)), [5.0])
    resized = resize_object_layers(obj, 4)

    numpy.testing.assert_allclose(list(resized.layer_spacing_m), [5.0, 5.0, 5.0], atol=1.0e-15)


def test_resize_object_layers_accepts_explicit_spacing() -> None:
    resized = resize_object_layers(_make_layered_object(2), 3, layer_spacing_m=[7.0, 8.0])

    numpy.testing.assert_allclose(list(resized.layer_spacing_m), [7.0, 8.0], atol=1.0e-15)


def test_resize_object_layers_rejects_a_nonpositive_layer_count() -> None:
    with pytest.raises(ValueError):
        resize_object_layers(_make_layered_object(3), 0)


def test_resize_object_layers_rejects_a_wrong_length_spacing() -> None:
    with pytest.raises(ValueError):
        resize_object_layers(_make_layered_object(2), 4, layer_spacing_m=[1.0])


# ---------------------------------------------------------------------------
# resample_object_layers
# ---------------------------------------------------------------------------


def test_resample_object_layers_identity_grid_round_trips() -> None:
    obj = _make_layered_object(6)
    resampled = resample_object_layers(obj, list(obj.layer_spacing_m))

    numpy.testing.assert_allclose(resampled.get_array(), obj.get_array(), atol=1.0e-12)


@pytest.mark.parametrize('num_layers', [2, 3, 6, 12, 20])
def test_resample_object_layers_preserves_the_layer_product(num_layers: int) -> None:
    """Conservation is structural: the interpolated cumulative path telescopes."""
    obj = _make_layered_object(6)
    resampled = resample_object_layers(
        obj, compute_uniform_layer_spacing_m(THICKNESS_M, num_layers)
    )

    assert resampled.num_layers == num_layers
    numpy.testing.assert_allclose(
        resampled.get_layers_flattened(), obj.get_layers_flattened(), atol=1.0e-12
    )


def test_resample_object_layers_round_trip_preserves_the_projected_object() -> None:
    obj = _make_layered_object(6)
    upsampled = resample_object_layers(obj, compute_uniform_layer_spacing_m(THICKNESS_M, 12))
    restored = resample_object_layers(upsampled, compute_uniform_layer_spacing_m(THICKNESS_M, 6))

    numpy.testing.assert_allclose(
        restored.get_layers_flattened(), obj.get_layers_flattened(), atol=1.0e-12
    )


def test_resample_object_layers_handles_a_total_phase_beyond_pi() -> None:
    obj = _make_constant_phase_object([2.0, 2.0, 2.0])
    resampled = resample_object_layers(obj, compute_uniform_layer_spacing_m(THICKNESS_M, 6))

    numpy.testing.assert_allclose(numpy.angle(resampled.get_array()).sum(axis=0), 6.0, atol=1.0e-10)


def test_resample_object_layers_single_source_layer_matches_homogenize() -> None:
    """With one layer there is no depth axis to resample; the operation is homogenization."""
    obj = _make_layered_object(1)
    spacing_m = compute_uniform_layer_spacing_m(THICKNESS_M, 5)

    numpy.testing.assert_allclose(
        resample_object_layers(obj, spacing_m).get_array(),
        homogenize_object_layers(obj, num_layers=5, layer_spacing_m=spacing_m).get_array(),
        atol=1.0e-15,
    )


def test_resample_object_layers_increments_do_not_reverse() -> None:
    """What the monotone interpolant buys: no output layer gets a phase step whose sign
    opposes a uniformly accumulating source."""
    obj = _make_constant_phase_object([0.4, 0.8, 1.2, 1.6])
    resampled = resample_object_layers(obj, compute_uniform_layer_spacing_m(THICKNESS_M, 11))

    assert numpy.all(numpy.angle(resampled.get_array()) > 0.0)


def test_resample_object_layers_tolerates_a_zero_amplitude_border() -> None:
    """pad_object pads with zeros, whose logarithm is -inf; the floor keeps it finite."""
    obj = _make_layered_object(4)
    padded = _make_object(
        numpy.pad(obj.get_array(), ((0, 0), (2, 2), (2, 2))), list(obj.layer_spacing_m)
    )
    resampled = resample_object_layers(padded, compute_uniform_layer_spacing_m(THICKNESS_M, 3))

    assert numpy.isfinite(resampled.get_array()).all()


def test_resample_object_layers_without_preservation_tracks_the_thickness() -> None:
    """Evaluated at their own depths, a thinner output stack transmits more."""
    obj = _make_layered_object(6)
    thinner_m = compute_uniform_layer_spacing_m(THICKNESS_M / 2.0, 6)
    resampled = resample_object_layers(obj, thinner_m, preserve_total_transmission=False)

    assert (
        numpy.abs(resampled.get_layers_flattened()).min()
        > numpy.abs(obj.get_layers_flattened()).min()
    )
