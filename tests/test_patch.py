"""Unit tests for probe-patch placement in ptychodus.api.patch."""

from __future__ import annotations

import numpy
import numpy.testing
import pytest

from ptychodus.api.fourier import fourier_shift_2d
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.interpolate import BarycentricArrayInterpolator
from ptychodus.api.object import Object, ObjectCenter, compute_object_geometry
from ptychodus.api.patch import (
    accumulate_probe_intensity,
    accumulate_probe_intensity_on,
    iter_probe_patches,
    iter_probe_patches_on,
)
from ptychodus.api.probe import ProbeGeometry, ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata

_PIXEL_SIZE_M = 1.0e-7
_PROBE_PX = 6


def _metadata() -> ProductMetadata:
    return ProductMetadata(
        name='patch',
        comments='',
        detector_distance_m=1.0,
        photon_energy_eV=10000.0,
        probe_photon_count=1.0e9,
        exposure_time_s=0.1,
        mass_attenuation_m2_per_kg=5.0,
        tomography_angle_deg=0.0,
    )


def _probe_array(num_modes: int = 1) -> numpy.ndarray:
    yy, xx = numpy.mgrid[:_PROBE_PX, :_PROBE_PX] - (_PROBE_PX - 1) / 2
    base = numpy.exp(-(yy**2 + xx**2) / 4.0).astype(complex)
    return numpy.stack([base / (mode + 1) for mode in range(num_modes)])


def _make_product(
    positions_px: list[tuple[float, float]],
    *,
    canvas_px: int = 24,
    num_modes: int = 1,
) -> Product:
    """Build a product whose scan positions sit at the given object-pixel centers."""
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    radius_px = (canvas_px - 1) / 2
    positions = [
        ProbePosition(
            index=index,
            x_m=(x_px - radius_px) * _PIXEL_SIZE_M,
            y_m=(y_px - radius_px) * _PIXEL_SIZE_M,
        )
        for index, (x_px, y_px) in enumerate(positions_px)
    ]
    rng = numpy.random.default_rng(5)
    array = rng.standard_normal((canvas_px, canvas_px)) + 1j * rng.standard_normal(
        (canvas_px, canvas_px)
    )

    return Product(
        metadata=_metadata(),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(
            array=_probe_array(num_modes), opr_weights=None, pixel_geometry=pixel_geometry
        ),
        object_=Object(
            array=array, pixel_geometry=pixel_geometry, center=ObjectCenter(x_m=0.0, y_m=0.0)
        ),
        losses=[],
    )


# A zero sub-pixel offset needs ``center - (extent - 1) / 2`` whole, so for an even
# extent the zero-offset centers are the half-integers.
_ZERO_OFFSET = [(8.5, 8.5), (11.5, 11.5), (14.5, 14.5)]
_SUBPIXEL = [(8.3, 9.7), (11.5, 11.5), (14.2, 13.8)]


def test_iter_probe_patches_yields_one_entry_per_position() -> None:
    product = _make_product(_SUBPIXEL)
    patches = list(iter_probe_patches(product))

    assert [patch.position.index for patch in patches] == [0, 1, 2]
    for patch in patches:
        assert patch.bounds.y_slice.stop - patch.bounds.y_slice.start == _PROBE_PX
        assert patch.modes.shape == (1, _PROBE_PX, _PROBE_PX)


def test_patch_intensity_matches_a_hand_computed_fourier_shift() -> None:
    product = _make_product(_SUBPIXEL, num_modes=3)

    for patch, (x_px, y_px) in zip(iter_probe_patches(product), _SUBPIXEL):
        half = (_PROBE_PX - 1) / 2
        expected_modes = fourier_shift_2d(
            _probe_array(3),
            dx=x_px - (int(x_px - half) + half),
            dy=y_px - (int(y_px - half) + half),
        )
        expected = numpy.sum(numpy.abs(expected_modes) ** 2, axis=0)
        numpy.testing.assert_allclose(patch.get_intensity(), expected, atol=1e-12)


def test_normalized_intensity_sums_to_one() -> None:
    for patch in iter_probe_patches(_make_product(_SUBPIXEL)):
        assert patch.get_normalized_intensity().sum() == pytest.approx(1.0)


def test_normalized_intensity_passes_through_an_all_zero_patch() -> None:
    product = _make_product(_ZERO_OFFSET)
    patch = next(iter_probe_patches(product))
    empty = type(patch)(
        position=patch.position, bounds=patch.bounds, modes=numpy.zeros_like(patch.modes)
    )

    numpy.testing.assert_array_equal(empty.get_normalized_intensity(), empty.get_intensity())


def test_iter_probe_patches_raises_for_a_position_off_the_canvas() -> None:
    """An unchecked negative lower bound is a valid Python slice that wraps around
    to the far edge of the canvas, so the position must be refused outright."""
    product = _make_product([(11.5, 11.5), (1.0, 1.0)])

    with pytest.raises(ValueError, match='scan index 1'):
        list(iter_probe_patches(product))


def test_canonical_canvas_admits_every_position() -> None:
    """A canvas sized from the scan holds every position's footprint, including the
    last row and column where the sub-pixel offset is exactly zero."""
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    step_m = 8.0 * _PIXEL_SIZE_M
    positions = [
        ProbePosition(index=row * 5 + column, x_m=column * step_m, y_m=row * step_m)
        for row in range(5)
        for column in range(5)
    ]
    probe_geometry = ProbeGeometry(
        width_px=_PROBE_PX,
        height_px=_PROBE_PX,
        pixel_width_m=_PIXEL_SIZE_M,
        pixel_height_m=_PIXEL_SIZE_M,
    )
    geometry = compute_object_geometry(positions, probe_geometry)
    product = Product(
        metadata=_metadata(),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(array=_probe_array(), opr_weights=None, pixel_geometry=pixel_geometry),
        object_=Object(
            array=numpy.ones((geometry.height_px, geometry.width_px), dtype=complex),
            pixel_geometry=pixel_geometry,
            center=geometry.get_center(),
        ),
        losses=[],
    )

    assert len(list(iter_probe_patches(product))) == len(positions)


def test_both_conventions_select_the_same_pixels_at_a_zero_offset() -> None:
    """With no sub-pixel offset the integer slice and the bilinear resample agree,
    which is what pins the two mechanisms to one centered-pixel convention."""
    product = _make_product(_ZERO_OFFSET)
    object_array = product.object_.get_layers_flattened()
    geometry = product.object_.get_geometry()

    for patch, (x_px, y_px) in zip(iter_probe_patches(product), _ZERO_OFFSET):
        sliced = object_array[patch.bounds.y_slice, patch.bounds.x_slice]
        resampled = BarycentricArrayInterpolator(object_array).get_patch(
            x_px, y_px, _PROBE_PX, _PROBE_PX
        )
        numpy.testing.assert_allclose(sliced, resampled, atol=1e-12)
        assert geometry.width_px >= patch.bounds.x_slice.stop


def test_accumulate_matches_the_per_position_loop() -> None:
    """The canvas-level entry point and the per-position one agree, so either can
    back the other."""
    product = _make_product(_SUBPIXEL)
    geometry = product.object_.get_geometry()

    expected = numpy.zeros((geometry.height_px, geometry.width_px))
    for patch in iter_probe_patches(product):
        expected[patch.bounds.y_slice, patch.bounds.x_slice] += patch.get_intensity()

    numpy.testing.assert_array_equal(accumulate_probe_intensity(product), expected)


def test_accumulate_applies_per_position_weights() -> None:
    product = _make_product(_SUBPIXEL)
    geometry = product.object_.get_geometry()
    weights = numpy.array([0.5, 2.0, 1.5])

    expected = numpy.zeros((geometry.height_px, geometry.width_px))
    for weight, patch in zip(weights, iter_probe_patches(product)):
        expected[patch.bounds.y_slice, patch.bounds.x_slice] += weight * patch.get_intensity()

    numpy.testing.assert_array_equal(accumulate_probe_intensity(product, weights=weights), expected)


def test_accumulate_with_stacked_weights_matches_separate_calls() -> None:
    """``K`` weight rows accumulate in one pass and agree with ``K`` single passes."""
    product = _make_product(_SUBPIXEL)
    rows = numpy.array([[0.5, 2.0, 1.5], [1.0, 1.0, 3.0]])

    stacked = accumulate_probe_intensity(product, weights=rows, normalize=True)

    assert stacked.shape == (2, *accumulate_probe_intensity(product).shape)
    for index, row in enumerate(rows):
        numpy.testing.assert_array_equal(
            stacked[index], accumulate_probe_intensity(product, weights=row, normalize=True)
        )


def test_accumulate_rejects_weights_of_the_wrong_length() -> None:
    product = _make_product(_SUBPIXEL)

    with pytest.raises(ValueError, match='position'):
        accumulate_probe_intensity(product, weights=numpy.ones(2))


def _make_opr_product(num_positions: int, opr_weights: numpy.ndarray) -> Product:
    """A product whose coherent modes are mixed per position by ``opr_weights``."""
    canvas_px = 24
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    radius_px = (canvas_px - 1) / 2
    positions = [
        ProbePosition(
            index=index,
            x_m=(10.5 + index - radius_px) * _PIXEL_SIZE_M,
            y_m=(11.5 - radius_px) * _PIXEL_SIZE_M,
        )
        for index in range(num_positions)
    ]
    num_coherent = opr_weights.shape[1]
    # (coherent, incoherent, height, width); __getitem__ mixes the coherent axis.
    array = numpy.stack([_probe_array(1)[0] * (mode + 1) for mode in range(num_coherent)])
    array = array[:, numpy.newaxis, :, :]

    return Product(
        metadata=_metadata(),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(array=array, opr_weights=opr_weights, pixel_geometry=pixel_geometry),
        object_=Object(
            array=numpy.ones((canvas_px, canvas_px), dtype=complex),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def test_iter_probe_patches_honors_opr_weights() -> None:
    """Each position carries its own coherent mixture, so the patches must differ.

    A patch iterator that reused one probe for the whole scan would pass every
    other test in this module, so this is the one that pins the contract.
    """
    weights = numpy.array([[1.0, 0.2], [0.4, 1.3], [0.9, 0.7]])
    product = _make_opr_product(3, weights)

    totals = [float(patch.get_intensity().sum()) for patch in iter_probe_patches(product)]

    assert len(totals) == 3
    assert len(set(numpy.round(totals, 9))) == 3


def test_accumulate_normalizes_each_position_separately_under_opr() -> None:
    """Normalization divides by the position's own total, which OPR makes vary."""
    weights = numpy.array([[1.0, 0.2], [0.4, 1.3], [0.9, 0.7]])
    product = _make_opr_product(3, weights)
    geometry = product.object_.get_geometry()

    expected = numpy.zeros((geometry.height_px, geometry.width_px))
    for patch in iter_probe_patches(product):
        intensity = patch.get_intensity()
        expected[patch.bounds.y_slice, patch.bounds.x_slice] += intensity / intensity.sum()

    numpy.testing.assert_array_equal(accumulate_probe_intensity(product, normalize=True), expected)


def test_fewer_opr_weight_rows_than_positions_is_rejected() -> None:
    """The shortfall is a disagreement between the probe and the scan, so it is
    reported as one -- naming both counts -- rather than surfacing as an array
    index error from inside the probe ensemble."""
    weights = numpy.array([[1.0, 0.2], [0.4, 1.3]])

    with pytest.raises(ValueError, match=r'2 probe position\(s\).*has 4'):
        list(iter_probe_patches(_make_opr_product(4, weights)))


def test_the_unpacked_forms_equal_the_product_forms() -> None:
    product = _make_product(_SUBPIXEL, num_modes=2)
    geometry = product.object_.get_geometry()

    by_product = list(iter_probe_patches(product))
    unpacked = list(iter_probe_patches_on(product.probes, product.probe_positions, geometry))

    assert [patch.bounds for patch in unpacked] == [patch.bounds for patch in by_product]
    for left, right in zip(unpacked, by_product):
        numpy.testing.assert_array_equal(left.modes, right.modes)

    weights = numpy.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    numpy.testing.assert_array_equal(
        accumulate_probe_intensity_on(
            product.probes, product.probe_positions, geometry, weights=weights
        ),
        accumulate_probe_intensity(product, weights=weights),
    )
