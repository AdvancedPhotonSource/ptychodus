"""Unit tests for the diffraction model: processor ops, sizer, and the settings → processor wiring.

These tests lock the intended behavior of `DiffractionPatternProcessor` and `PatternSizer`
so regressions in the op pipeline (crop, binning, padding, transpose, value filtering) and
the processed-extent math are caught at the unit level.
"""

import numpy
import pytest

from ptychodus.api.diffraction import CropCenter, SimpleDiffractionArray
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.settings import SettingsRegistry

from ptychodus.model.diffraction.processor import (
    DiffractionPatternBinning,
    DiffractionPatternCrop,
    DiffractionPatternFilterValues,
    DiffractionPatternPadding,
    DiffractionPatternProcessor,
)
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


def _zeros_patterns(shape: tuple[int, int, int]) -> numpy.ndarray:
    return numpy.zeros(shape, dtype=numpy.uint16)


# ---------- Filter ----------


def test_filter_lower_bound_zeros_below() -> None:
    data = numpy.array([[[0, 1, 5, 10]]], dtype=numpy.int32)
    out = DiffractionPatternFilterValues(lower_bound=3, upper_bound=None).apply(data.copy())
    assert out.tolist() == [[[0, 0, 5, 10]]]


def test_filter_upper_bound_zeros_at_or_above() -> None:
    data = numpy.array([[[0, 1, 5, 10]]], dtype=numpy.int32)
    out = DiffractionPatternFilterValues(lower_bound=None, upper_bound=5).apply(data.copy())
    assert out.tolist() == [[[0, 1, 0, 0]]]


def test_filter_does_not_mutate_input() -> None:
    """B5: filter must not scribble on the caller's buffer."""
    original = numpy.array([[[0, 1, 5, 10]]], dtype=numpy.int32)
    snapshot = original.copy()
    DiffractionPatternFilterValues(lower_bound=3, upper_bound=8).apply(original)
    assert numpy.array_equal(original, snapshot)


# ---------- Crop ----------


def test_crop_apply_reduces_shape_around_center() -> None:
    data = numpy.arange(2 * 8 * 8, dtype=numpy.uint16).reshape(2, 8, 8)
    crop = DiffractionPatternCrop(CropCenter(position_x_px=4, position_y_px=4), ImageExtent(4, 4))
    out = crop.apply(data)
    assert out.shape == (2, 4, 4)
    # Center-crop of 8x8 around (4,4) with radius 2 = rows 2:6, cols 2:6
    assert numpy.array_equal(out[0], data[0, 2:6, 2:6])


def test_crop_apply_mask_reduces_shape() -> None:
    data = numpy.ones((8, 8), dtype=bool)
    crop = DiffractionPatternCrop(CropCenter(position_x_px=4, position_y_px=4), ImageExtent(4, 4))
    assert crop.apply(data, is_mask=True).shape == (4, 4)


# ---------- Binning ----------


def test_binning_apply_sums_blocks() -> None:
    data = numpy.ones((1, 4, 4), dtype=numpy.uint16)
    out = DiffractionPatternBinning(bin_size_x=2, bin_size_y=2).apply(data)
    assert out.shape == (1, 2, 2)
    assert (out == 4).all()


def test_binning_apply_mask_logical_and() -> None:
    data = numpy.ones((4, 4), dtype=bool)
    data[0, 0] = False  # one True-cell of the (0,0) 2x2 block becomes False
    out = DiffractionPatternBinning(bin_size_x=2, bin_size_y=2).apply(data, is_mask=True)
    assert out.shape == (2, 2)
    assert out[0, 0] == False  # logical AND of the block
    assert out[0, 1] == True
    assert out[1, 0] == True
    assert out[1, 1] == True


# ---------- Padding (B1) ----------


def test_padding_apply_3d_produces_correct_shape() -> None:
    """B1: pad_width must broadcast to (ndim, 2); flat tuples raise ValueError."""
    data = _zeros_patterns((2, 4, 4))
    out = DiffractionPatternPadding(pad_x=1, pad_y=1).apply(data)
    assert out.shape == (2, 6, 6)
    assert (out == 0).all()


def test_padding_apply_mask_2d_produces_correct_shape() -> None:
    """B1 mirror: bad-pixels padding must not raise either."""
    data = numpy.ones((4, 4), dtype=bool)
    out = DiffractionPatternPadding(pad_x=1, pad_y=1).apply(data, is_mask=True)
    assert out.shape == (6, 6)
    # Edges are padded with False; interior preserved.
    assert out[0, 0] == False
    assert out[3, 3] == True


def test_padding_asymmetric_pad_x_pad_y() -> None:
    data = _zeros_patterns((1, 4, 6))
    out = DiffractionPatternPadding(pad_x=2, pad_y=1).apply(data)
    assert out.shape == (1, 6, 10)


# ---------- Processor.__call__ ----------


def _processor(
    *,
    crop: DiffractionPatternCrop | None = None,
    filter_values: DiffractionPatternFilterValues | None = None,
    binning: DiffractionPatternBinning | None = None,
    padding: DiffractionPatternPadding | None = None,
    hflip: bool = False,
    vflip: bool = False,
    transpose: bool = False,
) -> DiffractionPatternProcessor:
    return DiffractionPatternProcessor(
        crop=crop,
        filter_values=filter_values,
        binning=binning,
        padding=padding,
        hflip=hflip,
        vflip=vflip,
        transpose=transpose,
    )


def test_processor_promotes_2d_input_to_3d() -> None:
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((8, 8), dtype=numpy.uint16)
    )
    out = _processor()(array)
    assert out.get_patterns().shape == (1, 8, 8)


def test_processor_rejects_4d_input() -> None:
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 2, 4, 4), dtype=numpy.uint16)
    )
    with pytest.raises(ValueError, match='Invalid diffraction pattern dimensions'):
        _processor()(array)


def test_processor_padding_in_full_pipeline() -> None:
    """Padding inside a processor stack must succeed (regression for B1)."""
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.ones((1, 4, 4), dtype=numpy.uint16)
    )
    proc = _processor(padding=DiffractionPatternPadding(pad_x=1, pad_y=1))
    assert proc(array).get_patterns().shape == (1, 6, 6)


def test_processor_transpose_swaps_spatial_axes() -> None:
    patterns = numpy.zeros((1, 3, 5), dtype=numpy.uint16)
    array = SimpleDiffractionArray('a', numpy.zeros(1, dtype=int), patterns)
    assert _processor(transpose=True)(array).get_patterns().shape == (1, 5, 3)


# ---------- Processor.process_bad_pixels (B2) ----------


def test_process_bad_pixels_requires_2d() -> None:
    with pytest.raises(ValueError, match='Invalid bad_pixel dimensions'):
        _processor().process_bad_pixels(numpy.zeros((1, 4, 4), dtype=bool))


def test_process_bad_pixels_transpose_does_not_crash() -> None:
    """B2: transpose used axes=(0,2,1) on 2D, raising 'axes don't match array'."""
    bad = numpy.zeros((3, 5), dtype=bool)
    bad[0, 4] = True
    out = _processor(transpose=True).process_bad_pixels(bad)
    assert out.shape == (5, 3)
    assert out[4, 0] == True


def test_process_bad_pixels_padding_does_not_crash() -> None:
    """B1: padding flow on bad pixels must not raise."""
    bad = numpy.ones((4, 4), dtype=bool)
    out = _processor(padding=DiffractionPatternPadding(pad_x=1, pad_y=1)).process_bad_pixels(bad)
    assert out.shape == (6, 6)


def test_process_bad_pixels_full_pipeline() -> None:
    bad = numpy.zeros((8, 8), dtype=bool)
    bad[4, 4] = True
    out = _processor(
        crop=DiffractionPatternCrop(
            CropCenter(position_x_px=4, position_y_px=4), ImageExtent(4, 4)
        ),
        binning=DiffractionPatternBinning(bin_size_x=2, bin_size_y=2),
        padding=DiffractionPatternPadding(pad_x=1, pad_y=1),
    ).process_bad_pixels(bad)
    # 8x8 → crop to 4x4 (rows 2:6, cols 2:6, bad[4,4] inside) → bin 2x2 to 2x2 (logical AND so False) → pad to 4x4
    assert out.shape == (4, 4)


# ---------- Sizer (B3, B4) ----------


@pytest.fixture
def settings() -> tuple[DiffractionSettings, DetectorSettings]:
    reg = SettingsRegistry()
    return DiffractionSettings(reg), DetectorSettings(reg)


def test_sizer_processed_size_accounts_for_double_sided_padding(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """B4: padding is applied on both sides; processed size adds 2 * pad."""
    diff, det = settings
    det.width_px.set_value(64)
    det.height_px.set_value(64)
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(32)
    diff.crop_height_px.set_value(32)
    diff.crop_center_x_px.set_value(32)
    diff.crop_center_y_px.set_value(32)
    diff.binning_enabled.set_value(False)
    diff.padding_enabled.set_value(True)
    diff.pad_x.set_value(4)
    diff.pad_y.set_value(4)

    sizer = PatternSizer(det, diff)
    assert sizer.axis_x.get_processed_size() == 32 + 2 * 4
    assert sizer.axis_y.get_processed_size() == 32 + 2 * 4

    # Cross-check against the processor's actual output shape.
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 64, 64), dtype=numpy.uint16)
    )
    out_shape = sizer.get_processor()(array).get_patterns().shape
    assert out_shape == (1, sizer.axis_y.get_processed_size(), sizer.axis_x.get_processed_size())


def test_sizer_lower_bound_filter_uses_its_own_toggle(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """B3: enabling only value_lower_bound_enabled must activate the lower bound."""
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(True)
    diff.value_lower_bound.set_value(7)
    diff.value_upper_bound_enabled.set_value(False)

    proc = PatternSizer(det, diff).get_processor()
    assert proc.filter_values is not None
    assert proc.filter_values.lower_bound == 7
    assert proc.filter_values.upper_bound is None


def test_sizer_upper_bound_filter_uses_its_own_toggle(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(False)
    diff.value_upper_bound_enabled.set_value(True)
    diff.value_upper_bound.set_value(1234)

    proc = PatternSizer(det, diff).get_processor()
    assert proc.filter_values is not None
    assert proc.filter_values.lower_bound is None
    assert proc.filter_values.upper_bound == 1234


def test_sizer_both_filter_bounds_independent(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(True)
    diff.value_lower_bound.set_value(3)
    diff.value_upper_bound_enabled.set_value(True)
    diff.value_upper_bound.set_value(99)

    proc = PatternSizer(det, diff).get_processor()
    assert proc.filter_values is not None
    assert proc.filter_values.lower_bound == 3
    assert proc.filter_values.upper_bound == 99


def test_sizer_no_filter_bounds(settings: tuple[DiffractionSettings, DetectorSettings]) -> None:
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(False)
    diff.value_upper_bound_enabled.set_value(False)

    proc = PatternSizer(det, diff).get_processor()
    assert proc.filter_values is not None
    assert proc.filter_values.lower_bound is None
    assert proc.filter_values.upper_bound is None
