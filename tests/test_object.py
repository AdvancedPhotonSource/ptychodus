"""Tests for ptychodus.api.object.

Currently covers:
  - align_objects: sub-pixel alignment with probe-position-consistent center adjustment.
"""

from __future__ import annotations

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import (
    Object,
    ObjectCenter,
    ObjectGeometry,
    align_objects,
    compute_object_geometry,
)
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.probe_positions import ProbePosition


def _make_gaussian_feature(
    shape: tuple[int, int, int],
    center_yx: tuple[float, float],
    sigma_px: float = 1.5,
) -> numpy.ndarray:
    """Build a complex array with a single Gaussian feature for alignment tests."""
    _, height, width = shape
    y = numpy.arange(height)[:, None]
    x = numpy.arange(width)[None, :]
    cy, cx = center_yx
    amplitude = numpy.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma_px**2))
    array = numpy.zeros(shape, dtype=numpy.complex128)
    array[0] = amplitude.astype(numpy.complex128)
    return array


def _fft_shift_2d(array_2d: numpy.ndarray, shift_yx: tuple[float, float]) -> numpy.ndarray:
    """Apply a sub-pixel real-space shift via FFT phase ramp (matches align_objects)."""
    height, width = array_2d.shape
    ky = numpy.fft.fftfreq(height)
    kx = numpy.fft.fftfreq(width)
    ky_grid, kx_grid = numpy.meshgrid(ky, kx, indexing='ij')
    phase_ramp = numpy.exp(-2j * numpy.pi * (shift_yx[0] * ky_grid + shift_yx[1] * kx_grid))
    return numpy.fft.ifft2(numpy.fft.fft2(array_2d) * phase_ramp)


def test_align_objects_identity_returns_unchanged_array_and_center() -> None:
    rng = numpy.random.default_rng(42)
    array = (rng.standard_normal((1, 16, 16)) + 1j * rng.standard_normal((1, 16, 16))).astype(
        numpy.complex128
    )
    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=2.0e-9)
    center = ObjectCenter(coordinate_x_m=5.0e-6, coordinate_y_m=-3.0e-6)
    obj = Object(array=array, pixel_geometry=pixel_geom, center=center)

    cropped_reference, aligned = align_objects(obj, obj)

    assert cropped_reference.get_array().shape == array.shape
    numpy.testing.assert_allclose(cropped_reference.get_array(), array, atol=1.0e-15)
    numpy.testing.assert_allclose(
        aligned.get_center().coordinate_x_m, center.coordinate_x_m, atol=1.0e-15
    )
    numpy.testing.assert_allclose(
        aligned.get_center().coordinate_y_m, center.coordinate_y_m, atol=1.0e-15
    )
    numpy.testing.assert_allclose(aligned.get_array(), array, atol=1.0e-10)


def test_align_objects_integer_pixel_shift_y_axis_recovers_feature_position() -> None:
    """Moving has feature offset by +2 px in y; alignment must bring it back to reference."""
    reference_array = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 16.0))
    moving_array = _make_gaussian_feature((1, 32, 32), center_yx=(18.0, 16.0))
    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=2.0e-9)
    center = ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0)

    reference = Object(array=reference_array, pixel_geometry=pixel_geom, center=center)
    moving = Object(array=moving_array, pixel_geometry=pixel_geom, center=center)

    _, aligned = align_objects(reference, moving)
    aligned_amplitude = numpy.abs(aligned.get_array()[0])
    peak_idx = numpy.unravel_index(numpy.argmax(aligned_amplitude), aligned_amplitude.shape)

    assert peak_idx == (16, 16)
    # shift_yx = (-2, 0), so new_center.y = 0 - (-2) * 2e-9 = 4e-9
    numpy.testing.assert_allclose(aligned.get_center().coordinate_y_m, 4.0e-9, atol=1.0e-15)
    numpy.testing.assert_allclose(aligned.get_center().coordinate_x_m, 0.0, atol=1.0e-15)


def test_align_objects_integer_pixel_shift_x_axis_recovers_feature_position() -> None:
    """Same as the y-axis test but along x, with a different pixel pitch to catch axis swaps."""
    reference_array = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 16.0))
    moving_array = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 19.0))
    pixel_geom = PixelGeometry(width_m=3.0e-9, height_m=1.0e-9)
    center = ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0)

    reference = Object(array=reference_array, pixel_geometry=pixel_geom, center=center)
    moving = Object(array=moving_array, pixel_geometry=pixel_geom, center=center)

    _, aligned = align_objects(reference, moving)
    aligned_amplitude = numpy.abs(aligned.get_array()[0])
    peak_idx = numpy.unravel_index(numpy.argmax(aligned_amplitude), aligned_amplitude.shape)

    assert peak_idx == (16, 16)
    # shift_yx = (0, -3), so new_center.x = 0 - (-3) * 3e-9 = 9e-9
    numpy.testing.assert_allclose(aligned.get_center().coordinate_x_m, 9.0e-9, atol=1.0e-15)
    numpy.testing.assert_allclose(aligned.get_center().coordinate_y_m, 0.0, atol=1.0e-15)


def test_align_objects_subpixel_shift_recovered_within_upsample_tolerance() -> None:
    """A 1.5-px shift should be recovered to ~1/upsample_factor precision."""
    reference_array = _make_gaussian_feature((1, 64, 64), center_yx=(32.0, 32.0), sigma_px=3.0)
    moving_layer = _fft_shift_2d(reference_array[0], shift_yx=(1.5, -0.75))
    moving_array = moving_layer[numpy.newaxis]
    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=1.0e-9)
    center = ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0)

    reference = Object(array=reference_array, pixel_geometry=pixel_geom, center=center)
    moving = Object(array=moving_array, pixel_geometry=pixel_geom, center=center)

    upsample_factor = 100
    _, aligned = align_objects(reference, moving, upsample_factor=upsample_factor)

    # After alignment, the moving array should resemble the reference (up to FFT roundtrip noise).
    residual = numpy.abs(aligned.get_array()[0] - reference_array[0]).max()
    assert residual < 1.0e-3

    # Recovered center should agree with the analytical formula to within sub-pixel precision.
    # shift_yx = (-1.5, +0.75), so new_center = (0,0) - (-1.5, 0.75) * (height, width)
    expected_y_m = -(-1.5) * pixel_geom.height_m
    expected_x_m = -(0.75) * pixel_geom.width_m
    tolerance_m = pixel_geom.width_m / upsample_factor
    numpy.testing.assert_allclose(
        aligned.get_center().coordinate_y_m, expected_y_m, atol=tolerance_m
    )
    numpy.testing.assert_allclose(
        aligned.get_center().coordinate_x_m, expected_x_m, atol=tolerance_m
    )


def test_align_objects_probe_position_consistency_world_coordinate_invariant() -> None:
    """A probe position that addressed content in moving should still address it after alignment.

    This is the load-bearing invariant the new center logic provides.
    """
    moving_feature_yx = (20.0, 12.0)
    reference_feature_yx = (16.0, 16.0)
    moving_array = _make_gaussian_feature((1, 32, 32), center_yx=moving_feature_yx)
    reference_array = _make_gaussian_feature((1, 32, 32), center_yx=reference_feature_yx)
    pixel_geom = PixelGeometry(width_m=2.0e-9, height_m=3.0e-9)
    moving_center = ObjectCenter(coordinate_x_m=7.0e-9, coordinate_y_m=-5.0e-9)

    reference = Object(
        array=reference_array,
        pixel_geometry=pixel_geom,
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
    )
    moving = Object(array=moving_array, pixel_geometry=pixel_geom, center=moving_center)

    moving_geometry = moving.get_geometry()
    # World coord of the moving feature, by moving's own coordinate frame:
    rx_px = moving.width_px / 2
    ry_px = moving.height_px / 2
    feature_world_x = (
        moving_center.coordinate_x_m + (moving_feature_yx[1] - rx_px) * pixel_geom.width_m
    )
    feature_world_y = (
        moving_center.coordinate_y_m + (moving_feature_yx[0] - ry_px) * pixel_geom.height_m
    )

    probe_at_feature = ProbePosition(
        index=0, coordinate_x_m=feature_world_x, coordinate_y_m=feature_world_y
    )

    # Sanity check: in moving's frame, this probe maps to the feature pixel.
    pre_align_pos = moving_geometry.map_coordinates_probe_to_object(probe_at_feature)
    numpy.testing.assert_allclose(pre_align_pos.coordinate_y_px, moving_feature_yx[0], atol=1e-9)
    numpy.testing.assert_allclose(pre_align_pos.coordinate_x_px, moving_feature_yx[1], atol=1e-9)

    _, aligned = align_objects(reference, moving)
    aligned_geometry = aligned.get_geometry()
    post_align_pos = aligned_geometry.map_coordinates_probe_to_object(probe_at_feature)

    # The aligned array now has the feature at the reference's pixel location; the same probe
    # (in world coordinates) must map to that new pixel location.
    numpy.testing.assert_allclose(
        post_align_pos.coordinate_y_px, reference_feature_yx[0], atol=1e-9
    )
    numpy.testing.assert_allclose(
        post_align_pos.coordinate_x_px, reference_feature_yx[1], atol=1e-9
    )

    # And the aligned-array amplitude at that pixel should be near the peak of the moving feature
    # (i.e. the probe still "sees" the same physical content).
    aligned_amplitude = numpy.abs(aligned.get_array()[0])
    peak_value = aligned_amplitude.max()
    value_at_probe = aligned_amplitude[
        int(round(post_align_pos.coordinate_y_px)),
        int(round(post_align_pos.coordinate_x_px)),
    ]
    assert value_at_probe > 0.99 * peak_value


def test_align_objects_preserves_complex_phase() -> None:
    """The FFT phase-ramp shift must preserve complex phase content, not just amplitude."""
    rng = numpy.random.default_rng(7)
    height, width = 32, 32
    # Smooth complex pattern: amplitude AND non-trivial phase.
    y = numpy.arange(height)[:, None]
    x = numpy.arange(width)[None, :]
    amplitude = numpy.exp(-(((y - 16) ** 2 + (x - 16) ** 2) / 50.0))
    phase = 0.5 * numpy.sin(2 * numpy.pi * y / height) + 0.3 * numpy.cos(2 * numpy.pi * x / width)
    reference_layer = (amplitude * numpy.exp(1j * phase)).astype(numpy.complex128)
    shift_yx = (2.0, -3.0)  # integer pixels keeps interpolation noise low
    moving_layer = _fft_shift_2d(reference_layer, shift_yx=shift_yx)

    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=1.0e-9)
    center = ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0)
    reference = Object(
        array=reference_layer[numpy.newaxis], pixel_geometry=pixel_geom, center=center
    )
    moving = Object(array=moving_layer[numpy.newaxis], pixel_geometry=pixel_geom, center=center)

    _, aligned = align_objects(reference, moving)
    aligned_layer = aligned.get_array()[0]

    # Reference and aligned should match in both amplitude and phase across the support region.
    support = amplitude > 0.05
    numpy.testing.assert_allclose(
        numpy.abs(aligned_layer)[support], numpy.abs(reference_layer)[support], atol=1.0e-6
    )
    numpy.testing.assert_allclose(
        numpy.angle(aligned_layer)[support],
        numpy.angle(reference_layer)[support],
        atol=1.0e-5,
    )
    # silence unused-rng warning
    del rng


def test_align_objects_new_center_independent_of_reference_center() -> None:
    """Regression test: the aligned object's center must come from moving's center, not reference's.

    Previously the function set ``center=reference.get_center()``, which silently produced an
    internally inconsistent product whenever the two centers disagreed.
    """
    reference_array = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 16.0))
    moving_array = _make_gaussian_feature((1, 32, 32), center_yx=(18.0, 16.0))
    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=1.0e-9)

    reference_center = ObjectCenter(coordinate_x_m=100.0e-9, coordinate_y_m=200.0e-9)
    moving_center = ObjectCenter(coordinate_x_m=7.0e-9, coordinate_y_m=-5.0e-9)

    reference = Object(array=reference_array, pixel_geometry=pixel_geom, center=reference_center)
    moving = Object(array=moving_array, pixel_geometry=pixel_geom, center=moving_center)

    _, aligned = align_objects(reference, moving)
    # shift_yx = (-2, 0), so expected new_center = moving.center - shift_yx * pixel
    expected_y_m = moving_center.coordinate_y_m - (-2.0) * pixel_geom.height_m
    expected_x_m = moving_center.coordinate_x_m - 0.0 * pixel_geom.width_m

    numpy.testing.assert_allclose(aligned.get_center().coordinate_x_m, expected_x_m, atol=1.0e-15)
    numpy.testing.assert_allclose(aligned.get_center().coordinate_y_m, expected_y_m, atol=1.0e-15)
    # Crucially, the result is NOT just reference's center.
    assert aligned.get_center() != reference_center


def test_align_objects_preserves_multi_layer_count_and_spacing() -> None:
    """Multi-layer moving objects must round-trip layer count, layer_spacing_m, and per-layer shift."""
    reference_layer = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 16.0))[0]
    moving_layer_a = _make_gaussian_feature((1, 32, 32), center_yx=(18.0, 17.0))[0]
    moving_layer_b = _make_gaussian_feature((1, 32, 32), center_yx=(18.0, 17.0))[0] * (1.0 + 0.5j)

    reference_array = numpy.stack([reference_layer, reference_layer], axis=0)
    moving_array = numpy.stack([moving_layer_a, moving_layer_b], axis=0)

    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=1.0e-9)
    center = ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0)
    layer_spacing_m = [1.5e-6]

    reference = Object(
        array=reference_array,
        pixel_geometry=pixel_geom,
        center=center,
        layer_spacing_m=layer_spacing_m,
    )
    moving = Object(
        array=moving_array,
        pixel_geometry=pixel_geom,
        center=center,
        layer_spacing_m=layer_spacing_m,
    )

    _, aligned = align_objects(reference, moving)

    assert aligned.num_layers == 2
    assert list(aligned.layer_spacing_m) == layer_spacing_m

    # Both aligned layers should have their peak at (16, 16) — the same shift applied to both.
    for layer_idx in range(2):
        amplitude = numpy.abs(aligned.get_array()[layer_idx])
        peak = numpy.unravel_index(numpy.argmax(amplitude), amplitude.shape)
        assert peak == (16, 16)

    # Layer B's complex scale (1 + 0.5j) must be preserved.
    peak_b = aligned.get_array()[1, 16, 16]
    numpy.testing.assert_allclose(numpy.angle(peak_b), numpy.angle(1.0 + 0.5j), atol=1.0e-3)


def test_align_objects_raises_on_pixel_geometry_mismatch() -> None:
    array = _make_gaussian_feature((1, 16, 16), center_yx=(8.0, 8.0))
    center = ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0)
    reference = Object(
        array=array, pixel_geometry=PixelGeometry(width_m=1.0e-9, height_m=1.0e-9), center=center
    )
    moving = Object(
        array=array, pixel_geometry=PixelGeometry(width_m=2.0e-9, height_m=1.0e-9), center=center
    )
    with pytest.raises(ValueError, match='pixel geometry mismatch'):
        align_objects(reference, moving)


def test_align_objects_trims_to_common_shape_even_delta() -> None:
    """Even shape differences: both outputs share the min shape, reference center unchanged."""
    reference_array = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 16.0))
    moving_array = _make_gaussian_feature((1, 34, 36), center_yx=(17.0, 18.0))
    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=2.0e-9)
    reference_center = ObjectCenter(coordinate_x_m=5.0e-9, coordinate_y_m=-3.0e-9)
    moving_center = ObjectCenter(coordinate_x_m=5.0e-9, coordinate_y_m=-3.0e-9)

    reference = Object(array=reference_array, pixel_geometry=pixel_geom, center=reference_center)
    moving = Object(array=moving_array, pixel_geometry=pixel_geom, center=moving_center)

    cropped_reference, aligned_moving = align_objects(reference, moving)

    assert cropped_reference.get_array().shape == (1, 32, 32)
    assert aligned_moving.get_array().shape == (1, 32, 32)
    # Reference had no crop; center preserved exactly.
    numpy.testing.assert_allclose(
        cropped_reference.get_center().coordinate_x_m, reference_center.coordinate_x_m, atol=1.0e-15
    )
    numpy.testing.assert_allclose(
        cropped_reference.get_center().coordinate_y_m, reference_center.coordinate_y_m, atol=1.0e-15
    )
    # Moving had an even shape delta; the crop preserves its center exactly, then the sub-pixel
    # correlator adjusts it. Verify the aligned peak lands at reference's peak pixel.
    aligned_amplitude = numpy.abs(aligned_moving.get_array()[0])
    peak_idx = numpy.unravel_index(numpy.argmax(aligned_amplitude), aligned_amplitude.shape)
    assert peak_idx == (16, 16)


def test_align_objects_trims_to_common_shape_odd_delta_adjusts_center() -> None:
    """Odd shape difference: cropped center shifts by half a pixel on the odd axis."""
    reference_array = _make_gaussian_feature((1, 33, 32), center_yx=(16.0, 16.0))
    moving_array = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 16.0))
    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=2.0e-9)
    reference_center = ObjectCenter(coordinate_x_m=5.0e-9, coordinate_y_m=-3.0e-9)
    moving_center = ObjectCenter(coordinate_x_m=5.0e-9, coordinate_y_m=-3.0e-9)

    reference = Object(array=reference_array, pixel_geometry=pixel_geom, center=reference_center)
    moving = Object(array=moving_array, pixel_geometry=pixel_geom, center=moving_center)

    cropped_reference, _ = align_objects(reference, moving)

    # Reference height: 33 → 32, delta 1, h_start = 0; center shift = 0 - 0.5 = -0.5 px in y.
    expected_reference_y_m = reference_center.coordinate_y_m + (-0.5) * pixel_geom.height_m
    numpy.testing.assert_allclose(
        cropped_reference.get_center().coordinate_y_m, expected_reference_y_m, atol=1.0e-15
    )
    # x axis: delta 0, no shift.
    numpy.testing.assert_allclose(
        cropped_reference.get_center().coordinate_x_m, reference_center.coordinate_x_m, atol=1.0e-15
    )
    assert cropped_reference.get_array().shape == (1, 32, 32)


def test_align_objects_subpixel_alignment_survives_trimming() -> None:
    """A sub-pixel shift on a mismatched-shape pair is still recovered after trim.

    Builds moving as a Fourier-shifted copy of the reference (so the two arrays
    represent the same signal up to a known sub-pixel shift), embeds it in a
    slightly larger array to force the trim path, and checks that
    align_objects recovers the reference after center-cropping to the common
    shape and applying the sub-pixel Fourier shift.
    """
    reference_layer = _make_gaussian_feature((1, 32, 32), center_yx=(16.0, 16.0), sigma_px=3.0)[0]
    # Same content as reference, shifted sub-pixel; embed in a 34x34 array with a 1-pixel
    # border on every side so center-cropping to 32x32 recovers the shifted content exactly.
    shift_yx = (0.5, -0.25)
    shifted_layer = _fft_shift_2d(reference_layer, shift_yx=shift_yx)
    moving_layer_full = numpy.zeros((1, 34, 34), dtype=numpy.complex128)
    moving_layer_full[0, 1:33, 1:33] = shifted_layer

    pixel_geom = PixelGeometry(width_m=1.0e-9, height_m=1.0e-9)
    center = ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0)
    reference = Object(
        array=reference_layer[numpy.newaxis], pixel_geometry=pixel_geom, center=center
    )
    moving = Object(array=moving_layer_full, pixel_geometry=pixel_geom, center=center)

    upsample_factor = 100
    cropped_reference, aligned_moving = align_objects(
        reference, moving, upsample_factor=upsample_factor
    )

    assert cropped_reference.get_array().shape == (1, 32, 32)
    assert aligned_moving.get_array().shape == (1, 32, 32)

    # After trim + sub-pixel alignment, the aligned moving array must resemble the reference
    # in the support region (Fourier-shift wrap-around at the border is expected).
    support = numpy.abs(reference_layer) > 0.05
    residual = numpy.abs(aligned_moving.get_array()[0] - reference_layer)[support].max()
    assert residual < 5.0e-3


def _make_positions(*coords_xy_m: tuple[float, float]) -> list[ProbePosition]:
    return [
        ProbePosition(index=i, coordinate_x_m=x, coordinate_y_m=y)
        for i, (x, y) in enumerate(coords_xy_m)
    ]


def _make_probe_geometry(width_px: int = 64, pixel_m: float = 1.0e-8) -> ProbeGeometry:
    return ProbeGeometry(
        width_px=width_px,
        height_px=width_px,
        pixel_width_m=pixel_m,
        pixel_height_m=pixel_m,
    )


class TestComputeObjectGeometry:
    def test_empty_positions_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            compute_object_geometry([], _make_probe_geometry())

    def test_single_position_covers_probe_extent_at_that_point(self) -> None:
        probe = _make_probe_geometry(width_px=32, pixel_m=1.0e-8)
        positions = _make_positions((5.0e-7, -3.0e-7))

        geometry = compute_object_geometry(positions, probe)

        # A single scan point has zero bounding-box extent, so the object is exactly
        # the probe canvas centered on that point (rounded up by math.ceil).
        assert geometry.width_px == 32
        assert geometry.height_px == 32
        numpy.testing.assert_allclose(geometry.center_x_m, 5.0e-7)
        numpy.testing.assert_allclose(geometry.center_y_m, -3.0e-7)
        numpy.testing.assert_allclose(geometry.pixel_width_m, 1.0e-8)
        numpy.testing.assert_allclose(geometry.pixel_height_m, 1.0e-8)

    def test_bounding_box_arithmetic_expands_canvas_by_scan_span(self) -> None:
        probe = _make_probe_geometry(width_px=32, pixel_m=1.0e-8)
        # scan spans 200 nm in x, 100 nm in y around (100 nm, 50 nm)
        positions = _make_positions(
            (0.0, 0.0),
            (2.0e-7, 1.0e-7),
        )

        geometry = compute_object_geometry(positions, probe)

        # width_m = scan_span + probe_width = 2e-7 + 32 * 1e-8 = 5.2e-7 -> 52 px
        # height_m = scan_span + probe_height = 1e-7 + 32 * 1e-8 = 4.2e-7 -> 42 px
        assert geometry.width_px == 52
        assert geometry.height_px == 42
        numpy.testing.assert_allclose(geometry.center_x_m, 1.0e-7)
        numpy.testing.assert_allclose(geometry.center_y_m, 5.0e-8)

    def test_padding_widens_the_canvas_on_both_sides(self) -> None:
        probe = _make_probe_geometry(width_px=32, pixel_m=1.0e-8)
        positions = _make_positions((0.0, 0.0))

        no_padding = compute_object_geometry(positions, probe)
        with_padding = compute_object_geometry(positions, probe, padding_px=5)

        # 5 pixels per side on each axis; 10 pixels total per dimension.
        assert with_padding.width_px == no_padding.width_px + 10
        assert with_padding.height_px == no_padding.height_px + 10
        # Center is unchanged by symmetric padding.
        numpy.testing.assert_allclose(with_padding.center_x_m, no_padding.center_x_m)
        numpy.testing.assert_allclose(with_padding.center_y_m, no_padding.center_y_m)


class TestObjectGeometryStr:
    def test_renders_pixel_count_and_center_with_si_prefixes(self) -> None:
        geometry = ObjectGeometry(
            width_px=1024,
            height_px=1024,
            pixel_width_m=8.06e-9,
            pixel_height_m=8.06e-9,
            center_x_m=1.234e-6,
            center_y_m=-3.456e-6,
        )
        rendered = str(geometry)
        assert rendered.startswith('1024 x 1024 px around ')
        assert 'µm' in rendered  # SI prefix selected by magnitude
