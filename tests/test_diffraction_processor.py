"""Unit tests for the diffraction preprocessing pipeline and the settings → plan factory.

These tests lock the intended behavior of `DiffractionPrepPipeline` step ops and
`PrepPipelineBuilder.get_plan()` so regressions in the op chain (binning,
padding, transpose, value filtering), the processed-extent math, and the load-time
crop region are caught at the unit level.
"""

import numpy
import pytest

from ptychodus.api.diffraction import BeamCenter, CropRegion, SimpleDiffractionArray
from ptychodus.api.preprocess.diffraction import (
    BinningStep,
    DiffractionPrepPipeline,
    DiffractionPrepStepUnion,
    FilterValuesStep,
    HorizontalFlipStep,
    PaddingStep,
    TransposeStep,
    VerticalFlipStep,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.settings import SettingsRegistry

from ptychodus.model.diffraction.prep_pipeline import PrepPipelineBuilder
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings


def _zeros_patterns(shape: tuple[int, int, int]) -> numpy.ndarray:
    return numpy.zeros(shape, dtype=numpy.uint16)


# ---------- Filter ----------


def test_filter_lower_bound_zeros_below() -> None:
    data = numpy.array([[[0, 1, 5, 10]]], dtype=numpy.int32)
    out = FilterValuesStep(lower_bound=3, upper_bound=None).apply(data.copy())
    assert out.tolist() == [[[0, 0, 5, 10]]]


def test_filter_upper_bound_zeros_at_or_above() -> None:
    data = numpy.array([[[0, 1, 5, 10]]], dtype=numpy.int32)
    out = FilterValuesStep(lower_bound=None, upper_bound=5).apply(data.copy())
    assert out.tolist() == [[[0, 1, 0, 0]]]


def test_filter_does_not_mutate_input() -> None:
    """B5: filter must not scribble on the caller's buffer."""
    original = numpy.array([[[0, 1, 5, 10]]], dtype=numpy.int32)
    snapshot = original.copy()
    FilterValuesStep(lower_bound=3, upper_bound=8).apply(original)
    assert numpy.array_equal(original, snapshot)


def test_filter_is_noop_on_mask() -> None:
    """Value filtering is meaningless for boolean masks; step must short-circuit."""
    mask = numpy.array([[True, False], [True, True]], dtype=bool)
    out = FilterValuesStep(lower_bound=1, upper_bound=2).apply(mask)
    assert numpy.array_equal(out, mask)


# ---------- CropRegion.apply_to (replaces removed CropStep) ----------


def test_crop_region_apply_to_reduces_stack_shape_around_center() -> None:
    data = numpy.arange(2 * 8 * 8, dtype=numpy.uint16).reshape(2, 8, 8)
    region = CropRegion.from_center_extent(
        BeamCenter(x_px=4, y_px=4), ImageExtent(width_px=4, height_px=4)
    )
    out = region.apply_to(data)
    assert out.shape == (2, 4, 4)
    # Center-crop of 8x8 around (4,4) with radius 2 = rows 2:6, cols 2:6
    assert numpy.array_equal(out[0], data[0, 2:6, 2:6])


def test_crop_region_apply_to_reduces_2d_mask_shape() -> None:
    data = numpy.ones((8, 8), dtype=bool)
    region = CropRegion.from_center_extent(
        BeamCenter(x_px=4, y_px=4), ImageExtent(width_px=4, height_px=4)
    )
    assert region.apply_to(data).shape == (4, 4)


# ---------- Binning ----------


def test_binning_apply_sums_blocks() -> None:
    data = numpy.ones((1, 4, 4), dtype=numpy.uint16)
    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)
    assert out.shape == (1, 2, 2)
    assert (out == 4).all()


def test_binning_apply_mask_logical_or() -> None:
    """One bad sub-pixel makes the whole binned pixel bad.

    Bad pixels are zeroed before binning, so the block's sum is an undercount that
    must not pass as a measurement.
    """
    data = numpy.zeros((4, 4), dtype=bool)
    data[0, 0] = True  # one bad cell in the (0,0) 2x2 block
    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)
    assert out.shape == (2, 2)
    assert out[0, 0] == True
    assert out[0, 1] == False
    assert out[1, 0] == False
    assert out[1, 1] == False


def test_binning_apply_drops_the_trailing_partial_block() -> None:
    """An extent the bin size does not divide leaves a partial block at the far edge.

    It is discarded rather than refused, which is the contract apply_to_extent has
    always advertised with its floor division.
    """
    data = numpy.arange(5 * 5, dtype=numpy.uint16).reshape(1, 5, 5)

    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)

    assert out.shape == (1, 2, 2)
    # Row 4 and column 4 take no part; each output cell is the sum of its 2x2 block.
    expected = numpy.array(
        [[[0 + 1 + 5 + 6, 2 + 3 + 7 + 8], [10 + 11 + 15 + 16, 12 + 13 + 17 + 18]]]
    )
    numpy.testing.assert_array_equal(out, expected)


def test_binning_apply_shape_agrees_with_apply_to_extent() -> None:
    """The two halves of the step must report the same output size, divisible or not."""
    step = BinningStep(bin_size_x=2, bin_size_y=4)

    for height, width in ((12, 8), (9, 12), (13, 7), (5, 5)):
        out = step.apply(numpy.ones((1, height, width), dtype=numpy.uint16))
        extent = step.apply_to_extent(ImageExtent(width, height))
        assert out.shape[-2:] == (extent.height_px, extent.width_px)


def test_binning_apply_mask_drops_the_trailing_partial_block() -> None:
    """The discarded edge must not influence the binned mask."""
    data = numpy.zeros((5, 5), dtype=bool)
    data[4, 4] = True  # in the dropped row and column only

    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)

    assert out.shape == (2, 2)
    assert not out.any()


def test_binning_rejects_zero_bin_size() -> None:
    with pytest.raises(ValueError):
        BinningStep(bin_size_x=0, bin_size_y=1)


def test_binning_apply_promotes_dtype_to_avoid_overflow() -> None:
    """A saturated uint16 frame binned 2x2 must not wrap: 65535*4 = 262140 exceeds uint16."""
    data = numpy.full((1, 4, 4), 65535, dtype=numpy.uint16)
    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)
    assert out.shape == (1, 2, 2)
    assert numpy.issubdtype(out.dtype, numpy.unsignedinteger)
    assert numpy.iinfo(out.dtype).max >= 65535 * 4
    assert (out == 65535 * 4).all()


def test_binning_apply_preserves_dtype_when_bin_is_one() -> None:
    """A 1x1 bin is a no-op on values; the dtype must not be widened."""
    data = numpy.ones((1, 4, 4), dtype=numpy.uint16)
    out = BinningStep(bin_size_x=1, bin_size_y=1).apply(data)
    assert out.dtype == numpy.dtype(numpy.uint16)


def test_binning_apply_preserves_float_dtype() -> None:
    """Floats have the headroom; the promotion path must not force a widening."""
    data = numpy.ones((1, 4, 4), dtype=numpy.float32)
    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)
    assert out.dtype == numpy.dtype(numpy.float32)


# ---------- Padding (B1) ----------


def test_padding_apply_3d_produces_correct_shape() -> None:
    """B1: pad_width must broadcast to (ndim, 2); flat tuples raise ValueError."""
    data = _zeros_patterns((2, 4, 4))
    out = PaddingStep(pad_x=1, pad_y=1).apply(data)
    assert out.shape == (2, 6, 6)
    assert (out == 0).all()


def test_padding_apply_mask_2d_produces_correct_shape() -> None:
    """B1 mirror: bad-pixels padding must not raise either."""
    data = numpy.zeros((4, 4), dtype=bool)
    out = PaddingStep(pad_x=1, pad_y=1).apply(data)
    assert out.shape == (6, 6)
    # Padded pixels were never measured, so they are bad; the interior is preserved.
    assert out[0, 0] == True
    assert out[3, 3] == False
    assert out[1:-1, 1:-1].sum() == 0
    assert out.sum() == 6 * 6 - 4 * 4


def test_padding_asymmetric_pad_x_pad_y() -> None:
    data = _zeros_patterns((1, 4, 6))
    out = PaddingStep(pad_x=2, pad_y=1).apply(data)
    assert out.shape == (1, 6, 10)


def test_padding_rejects_negative() -> None:
    with pytest.raises(ValueError):
        PaddingStep(pad_x=-1, pad_y=0)


# ---------- Pipeline.__call__ ----------


def _pipeline(*steps: DiffractionPrepStepUnion) -> DiffractionPrepPipeline:
    return DiffractionPrepPipeline(steps=steps)


def test_pipeline_promotes_2d_input_to_3d() -> None:
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((8, 8), dtype=numpy.uint16)
    )
    out = _pipeline()(array)
    assert out.get_patterns().shape == (1, 8, 8)


def test_pipeline_rejects_4d_input() -> None:
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 2, 4, 4), dtype=numpy.uint16)
    )
    with pytest.raises(ValueError, match='Invalid diffraction pattern dimensions'):
        _pipeline()(array)


def test_pipeline_padding_in_full_pipeline() -> None:
    """Padding inside a pipeline stack must succeed (regression for B1)."""
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.ones((1, 4, 4), dtype=numpy.uint16)
    )
    assert _pipeline(PaddingStep(pad_x=1, pad_y=1))(array).get_patterns().shape == (1, 6, 6)


def test_pipeline_transpose_swaps_spatial_axes() -> None:
    patterns = numpy.zeros((1, 3, 5), dtype=numpy.uint16)
    array = SimpleDiffractionArray('a', numpy.zeros(1, dtype=int), patterns)
    assert _pipeline(TransposeStep())(array).get_patterns().shape == (1, 5, 3)


def test_pipeline_hflip_flips_last_axis() -> None:
    patterns = numpy.arange(6, dtype=numpy.uint16).reshape(1, 2, 3)
    array = SimpleDiffractionArray('a', numpy.zeros(1, dtype=int), patterns)
    out = _pipeline(HorizontalFlipStep())(array).get_patterns()
    assert numpy.array_equal(out[0], numpy.flip(patterns[0], axis=-1))


def test_pipeline_vflip_flips_second_to_last_axis() -> None:
    patterns = numpy.arange(6, dtype=numpy.uint16).reshape(1, 2, 3)
    array = SimpleDiffractionArray('a', numpy.zeros(1, dtype=int), patterns)
    out = _pipeline(VerticalFlipStep())(array).get_patterns()
    assert numpy.array_equal(out[0], numpy.flip(patterns[0], axis=-2))


# ---------- Pipeline.apply_to_mask (B2) ----------


def test_apply_to_mask_requires_2d() -> None:
    with pytest.raises(ValueError, match='Invalid bad_pixel dimensions'):
        _pipeline().apply_to_mask(numpy.zeros((1, 4, 4), dtype=bool))


def test_apply_to_mask_transpose_does_not_crash() -> None:
    """B2: transpose used axes=(0,2,1) on 2D, raising 'axes don't match array'."""
    bad = numpy.zeros((3, 5), dtype=bool)
    bad[0, 4] = True
    out = _pipeline(TransposeStep()).apply_to_mask(bad)
    assert out.shape == (5, 3)
    assert out[4, 0] == True


def test_apply_to_mask_padding_does_not_crash() -> None:
    """B1: padding flow on bad pixels must not raise."""
    bad = numpy.ones((4, 4), dtype=bool)
    out = _pipeline(PaddingStep(pad_x=1, pad_y=1)).apply_to_mask(bad)
    assert out.shape == (6, 6)


def test_apply_to_mask_full_pipeline_with_load_time_crop() -> None:
    """The pipeline no longer includes a crop step; callers crop the mask up front
    with `CropRegion.apply_to`, then run the pipeline (bin, pad, ...) on the
    cropped mask."""
    bad = numpy.zeros((8, 8), dtype=bool)
    bad[4, 4] = True
    region = CropRegion.from_center_extent(
        BeamCenter(x_px=4, y_px=4), ImageExtent(width_px=4, height_px=4)
    )
    out = _pipeline(
        BinningStep(bin_size_x=2, bin_size_y=2),
        PaddingStep(pad_x=1, pad_y=1),
    ).apply_to_mask(region.apply_to(bad))
    # 8x8 → crop to 4x4 (rows 2:6, cols 2:6, bad[4,4] inside) → bin 2x2 to 2x2 (logical OR, so
    # the block holding bad[4,4] is bad) → pad to 4x4 with a bad border
    assert out.shape == (4, 4)
    expected = numpy.ones((4, 4), dtype=bool)
    expected[1:3, 1:3] = False
    expected[2, 2] = True
    numpy.testing.assert_array_equal(out, expected)


# ---------- Step extent / pixel-geometry ----------


def test_binning_apply_to_extent_floor_divides() -> None:
    step = BinningStep(bin_size_x=2, bin_size_y=4)
    out = step.apply_to_extent(ImageExtent(9, 12))
    assert (out.width_px, out.height_px) == (4, 3)


def test_binning_apply_to_pixel_geometry_multiplies() -> None:
    step = BinningStep(bin_size_x=2, bin_size_y=4)
    out = step.apply_to_pixel_geometry(PixelGeometry(width_m=1e-5, height_m=2e-5))
    assert out.width_m == 2e-5
    assert out.height_m == 8e-5


def test_padding_apply_to_extent_adds_double_pad() -> None:
    step = PaddingStep(pad_x=1, pad_y=2)
    out = step.apply_to_extent(ImageExtent(4, 4))
    assert (out.width_px, out.height_px) == (6, 8)


def test_transpose_apply_to_extent_swaps_dimensions() -> None:
    step = TransposeStep()
    out = step.apply_to_extent(ImageExtent(3, 5))
    assert (out.width_px, out.height_px) == (5, 3)


def test_transpose_apply_to_pixel_geometry_swaps() -> None:
    step = TransposeStep()
    out = step.apply_to_pixel_geometry(PixelGeometry(width_m=1e-5, height_m=2e-5))
    assert out.width_m == 2e-5
    assert out.height_m == 1e-5


def test_identity_steps_do_not_change_extent_or_geometry() -> None:
    extent = ImageExtent(4, 6)
    geometry = PixelGeometry(width_m=1e-5, height_m=2e-5)
    for step in (
        FilterValuesStep(lower_bound=1, upper_bound=99),
        HorizontalFlipStep(),
        VerticalFlipStep(),
    ):
        assert step.apply_to_extent(extent) == extent
        assert step.apply_to_pixel_geometry(geometry) == geometry


def test_pipeline_compute_output_extent_composes_all_shape_steps() -> None:
    """With crop hoisted to load time, the pipeline sees an already-cropped input;
    compute_output_extent folds only the remaining shape-affecting steps."""
    pipeline = _pipeline(
        BinningStep(bin_size_x=2, bin_size_y=2),
        PaddingStep(pad_x=1, pad_y=1),
        TransposeStep(),
    )
    # 16x16 (already-cropped input) → bin 2x2 → 8x8 → pad → 10x10 → transpose → 10x10
    assert pipeline.compute_output_extent(ImageExtent(16, 16)) == ImageExtent(10, 10)


def test_pipeline_compute_output_pixel_geometry_composes_binning_and_transpose() -> None:
    pipeline = _pipeline(
        BinningStep(bin_size_x=2, bin_size_y=4),
        TransposeStep(),
    )
    out = pipeline.compute_output_pixel_geometry(PixelGeometry(width_m=1e-5, height_m=2e-5))
    # bin: width 2e-5, height 8e-5; transpose swaps → width 8e-5, height 2e-5
    assert out.width_m == 8e-5
    assert out.height_m == 2e-5


def test_default_apply_to_dtype_is_identity() -> None:
    """Steps that don't override apply_to_dtype must pass the dtype through unchanged."""
    assert TransposeStep().apply_to_dtype(numpy.dtype(numpy.uint16)) == numpy.dtype(numpy.uint16)


def test_pipeline_compute_output_dtype_folds_over_steps() -> None:
    """The pipeline fold must promote via BinningStep and identity-through TransposeStep."""
    pipeline = _pipeline(
        BinningStep(bin_size_x=4, bin_size_y=4),
        TransposeStep(),
    )
    out = pipeline.compute_output_dtype(numpy.dtype(numpy.uint16))
    assert numpy.issubdtype(out, numpy.unsignedinteger)
    assert numpy.iinfo(out).max >= 65535 * 16


# ---------- Serialization ----------


def test_pipeline_serializes_and_round_trips() -> None:
    """Pydantic tagged-union round-trip so `ptychodus_store` can persist a pipeline."""
    original = _pipeline(
        FilterValuesStep(lower_bound=1, upper_bound=99),
        BinningStep(bin_size_x=2, bin_size_y=2),
        PaddingStep(pad_x=1, pad_y=1),
        HorizontalFlipStep(),
        VerticalFlipStep(),
        TransposeStep(),
    )
    round_tripped = DiffractionPrepPipeline.model_validate_json(original.model_dump_json())
    assert round_tripped == original


# ---------- Sizer (B3, B4) ----------


@pytest.fixture
def settings() -> tuple[DiffractionSettings, DetectorSettings]:
    reg = SettingsRegistry()
    return DiffractionSettings(reg), DetectorSettings(reg)


def test_sizer_processed_size_accounts_for_double_sided_padding(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """B4: padding is applied on both sides; processed size adds 2 * pad.

    Crop is hoisted to a load-time read_region, so the pipeline sees the
    cropped input and compute_output_extent is folded from the region extent.
    """
    diff, det = settings
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(32)
    diff.crop_height_px.set_value(32)
    diff.beam_center_x_px.set_value(32)
    diff.beam_center_y_px.set_value(32)
    diff.binning_enabled.set_value(False)
    diff.padding_enabled.set_value(True)
    diff.pad_x.set_value(4)
    diff.pad_y.set_value(4)

    detector_extent = ImageExtent(width_px=64, height_px=64)
    plan = PrepPipelineBuilder(diff).get_plan(detector_extent)
    assert plan.read_region is not None
    cropped_extent = ImageExtent(
        width_px=plan.read_region.width_px, height_px=plan.read_region.height_px
    )
    extent = plan.pipeline.compute_output_extent(cropped_extent)
    assert extent.width_px == 32 + 2 * 4
    assert extent.height_px == 32 + 2 * 4

    # Cross-check against the pipeline's actual output shape on the cropped input.
    array = SimpleDiffractionArray(
        'a',
        numpy.zeros(1, dtype=int),
        numpy.zeros((1, cropped_extent.height_px, cropped_extent.width_px), dtype=numpy.uint16),
    )
    out_shape = plan.pipeline(array).get_patterns().shape
    assert out_shape == (1, extent.height_px, extent.width_px)


def test_sizer_processed_extent_reflects_transpose(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """Regression: transpose must swap width/height in the processed extent + pixel geometry."""
    diff, _ = settings
    diff.crop_enabled.set_value(False)
    diff.transpose.set_value(True)

    detector_extent = ImageExtent(width_px=64, height_px=32)
    plan = PrepPipelineBuilder(diff).get_plan(detector_extent)
    assert plan.read_region is None
    extent = plan.pipeline.compute_output_extent(detector_extent)
    assert (extent.width_px, extent.height_px) == (32, 64)

    geo = plan.pipeline.compute_output_pixel_geometry(PixelGeometry(width_m=1e-5, height_m=2e-5))
    assert (geo.width_m, geo.height_m) == (2e-5, 1e-5)

    # Cross-check: pipeline's actual output stack shape agrees with compute_output_extent.
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 32, 64), dtype=numpy.uint16)
    )
    out_shape = plan.pipeline(array).get_patterns().shape
    assert out_shape == (1, extent.height_px, extent.width_px)


def _filter_step(pipeline: DiffractionPrepPipeline) -> FilterValuesStep | None:
    for step in pipeline.steps:
        if isinstance(step, FilterValuesStep):
            return step
    return None


def test_sizer_lower_bound_filter_uses_its_own_toggle(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """B3: enabling only value_lower_bound_enabled must activate the lower bound."""
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(True)
    diff.value_lower_bound.set_value(7)
    diff.value_upper_bound_enabled.set_value(False)

    step = _filter_step(PrepPipelineBuilder(diff).get_plan().pipeline)
    assert step is not None
    assert step.lower_bound == 7
    assert step.upper_bound is None


def test_sizer_upper_bound_filter_uses_its_own_toggle(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(False)
    diff.value_upper_bound_enabled.set_value(True)
    diff.value_upper_bound.set_value(1234)

    step = _filter_step(PrepPipelineBuilder(diff).get_plan().pipeline)
    assert step is not None
    assert step.lower_bound is None
    assert step.upper_bound == 1234


def test_sizer_both_filter_bounds_independent(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(True)
    diff.value_lower_bound.set_value(3)
    diff.value_upper_bound_enabled.set_value(True)
    diff.value_upper_bound.set_value(99)

    step = _filter_step(PrepPipelineBuilder(diff).get_plan().pipeline)
    assert step is not None
    assert step.lower_bound == 3
    assert step.upper_bound == 99


def test_sizer_no_filter_bounds(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """Both filter toggles off → no FilterValuesStep in the pipeline (avoid a no-op step)."""
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(False)
    diff.value_upper_bound_enabled.set_value(False)

    assert _filter_step(PrepPipelineBuilder(diff).get_plan().pipeline) is None


# ---------- Load-time crop region ----------


@pytest.mark.parametrize(
    'user_center, expected',
    [
        (6, 6),  # max valid center (previous code clamped to 5)
        (2, 2),  # min valid center
        (0, 2),  # below min → clamped up to radius
        (100, 6),  # above max → clamped down to det_size - radius
        (4, 4),  # in-range identity
    ],
)
def test_sizer_safe_beam_center_matches_radius_bounds(
    settings: tuple[DiffractionSettings, DetectorSettings],
    user_center: int,
    expected: int,
) -> None:
    """The clamped center on the load-time read_region must equal the radius-based bounds."""
    diff, det = settings
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(4)
    diff.crop_height_px.set_value(4)
    diff.beam_center_x_px.set_value(user_center)
    diff.beam_center_y_px.set_value(user_center)

    detector_extent = ImageExtent(width_px=8, height_px=8)
    plan = PrepPipelineBuilder(diff).get_plan(detector_extent)
    assert plan.read_region is not None
    assert plan.read_region.center_x_px == expected
    assert plan.read_region.center_y_px == expected


def test_sizer_safe_beam_center_produces_in_bounds_slice(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """Cross-check: a max-valid center must yield a load-time slice fully inside the detector."""
    diff, det = settings
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(4)
    diff.crop_height_px.set_value(4)
    diff.beam_center_x_px.set_value(6)  # max valid; radius = 2 → slice [4:8]
    diff.beam_center_y_px.set_value(6)

    detector_extent = ImageExtent(width_px=8, height_px=8)
    plan = PrepPipelineBuilder(diff).get_plan(detector_extent)
    assert plan.read_region is not None
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 8, 8), dtype=numpy.uint16)
    )
    out = plan.pipeline(
        SimpleDiffractionArray(
            array.get_label(),
            array.get_indexes(),
            array.get_patterns(read_region=plan.read_region),
        )
    ).get_patterns()
    assert out.shape == (1, 4, 4)  # no truncation from an out-of-bounds slice


def test_get_plan_warns_when_crop_region_is_clamped(
    settings: tuple[DiffractionSettings, DetectorSettings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A crop window that overhangs the detector is a misconfiguration; say so loudly.

    Regression guard for the fly001 case: a stale settings key left the beam center at
    its default, so the requested window fell off the detector and was silently shifted
    onto a signal-free corner.
    """
    diff, _ = settings
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(128)
    diff.crop_height_px.set_value(128)
    diff.beam_center_x_px.set_value(32)
    diff.beam_center_y_px.set_value(32)

    detector_extent = ImageExtent(width_px=1030, height_px=514)

    with caplog.at_level('WARNING', logger='ptychodus.model.diffraction.prep_pipeline'):
        plan = PrepPipelineBuilder(diff).get_plan(detector_extent)

    assert plan.read_region == CropRegion(x_range=(0, 128), y_range=(0, 128))
    assert len(caplog.records) == 1
    assert 'BeamCenterXInPixels' in caplog.records[0].message


def test_get_plan_is_quiet_when_crop_region_fits(
    settings: tuple[DiffractionSettings, DetectorSettings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A crop that fits must not warn -- a warning that always fires stops being read."""
    diff, _ = settings
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(128)
    diff.crop_height_px.set_value(128)
    diff.beam_center_x_px.set_value(540)
    diff.beam_center_y_px.set_value(259)

    detector_extent = ImageExtent(width_px=1030, height_px=514)

    with caplog.at_level('WARNING', logger='ptychodus.model.diffraction.prep_pipeline'):
        plan = PrepPipelineBuilder(diff).get_plan(detector_extent)

    assert plan.read_region == CropRegion(x_range=(476, 604), y_range=(195, 323))
    assert not caplog.records


def test_get_plan_accepts_a_bin_size_that_does_not_divide_the_crop(
    settings: tuple[DiffractionSettings, DetectorSettings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-divisible bin size is a trim, not a misconfiguration.

    It used to raise, which aborted a batch run with an uncaught traceback and left
    the streaming path silently dropping every frame. The partial block is discarded
    and the loss is reported once per plan instead.
    """
    diff, _ = settings
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(54)
    diff.crop_height_px.set_value(54)
    diff.beam_center_x_px.set_value(64)
    diff.beam_center_y_px.set_value(64)
    diff.binning_enabled.set_value(True)
    diff.bin_size_x.set_value(4)
    diff.bin_size_y.set_value(4)

    detector_extent = ImageExtent(width_px=128, height_px=128)

    with caplog.at_level('WARNING', logger='ptychodus.model.diffraction.prep_pipeline'):
        plan = PrepPipelineBuilder(diff).get_plan(detector_extent)

    binning = next(step for step in plan.pipeline.steps if isinstance(step, BinningStep))
    assert (binning.bin_size_x, binning.bin_size_y) == (4, 4)
    assert len(caplog.records) == 1
    assert '2 column(s)' in caplog.records[0].message
    assert '2 row(s)' in caplog.records[0].message


def test_get_plan_is_quiet_when_the_bin_size_divides_the_crop(
    settings: tuple[DiffractionSettings, DetectorSettings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    diff, _ = settings
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(64)
    diff.crop_height_px.set_value(64)
    diff.beam_center_x_px.set_value(64)
    diff.beam_center_y_px.set_value(64)
    diff.binning_enabled.set_value(True)
    diff.bin_size_x.set_value(4)
    diff.bin_size_y.set_value(4)

    with caplog.at_level('WARNING', logger='ptychodus.model.diffraction.prep_pipeline'):
        PrepPipelineBuilder(diff).get_plan(ImageExtent(width_px=128, height_px=128))

    assert not caplog.records


# ---------- Additive per-pixel maps ----------


def test_pixel_map_binning_sums_in_float64() -> None:
    background = numpy.arange(16, dtype=numpy.float32).reshape(4, 4)

    out = _pipeline(BinningStep(bin_size_x=2, bin_size_y=2)).apply_to_pixel_map(background)

    assert out.dtype == numpy.float64
    numpy.testing.assert_array_equal(
        out, [[0 + 1 + 4 + 5, 2 + 3 + 6 + 7], [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]]
    )


def test_pixel_map_padding_fills_zero() -> None:
    out = _pipeline(PaddingStep(pad_x=1, pad_y=2)).apply_to_pixel_map(numpy.ones((2, 2)))

    assert out.shape == (6, 4)
    assert out.sum() == 4.0
    assert out[0, 0] == 0.0


def test_pixel_map_value_filter_is_a_noop() -> None:
    """A filter on measured values says nothing about a background's values."""
    background = numpy.full((2, 2), 1.0e6)
    step = FilterValuesStep(lower_bound=0, upper_bound=10)

    numpy.testing.assert_array_equal(_pipeline(step).apply_to_pixel_map(background), background)


def test_pixel_map_follows_flips_and_transposes() -> None:
    pixel_map = numpy.arange(6, dtype=float).reshape(2, 3)
    pipeline = _pipeline(HorizontalFlipStep(), VerticalFlipStep(), TransposeStep())

    numpy.testing.assert_array_equal(
        pipeline.apply_to_pixel_map(pixel_map), numpy.flip(pixel_map).T
    )


def test_pixel_map_must_be_2d() -> None:
    with pytest.raises(ValueError, match='pixel map'):
        _pipeline().apply_to_pixel_map(numpy.zeros((1, 2, 2)))
