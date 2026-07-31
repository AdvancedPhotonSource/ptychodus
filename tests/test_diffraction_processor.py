"""Unit tests for the diffraction preprocessing pipeline and the settings → pipeline factory.

These tests lock the intended behavior of `DiffractionPrepPipeline` step ops and
`PatternSizer.get_prep_pipeline()` so regressions in the op chain (crop, binning, padding,
transpose, value filtering) and the processed-extent math are caught at the unit level.
"""

import numpy
import pytest

from ptychodus.api.diffraction import CropCenter, SimpleDiffractionArray
from ptychodus.api.diffraction_prep import (
    BinningStep,
    CropStep,
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

from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


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


# ---------- Crop ----------


def test_crop_apply_reduces_shape_around_center() -> None:
    data = numpy.arange(2 * 8 * 8, dtype=numpy.uint16).reshape(2, 8, 8)
    crop = CropStep(center=CropCenter(position_x_px=4, position_y_px=4), extent=ImageExtent(4, 4))
    out = crop.apply(data)
    assert out.shape == (2, 4, 4)
    # Center-crop of 8x8 around (4,4) with radius 2 = rows 2:6, cols 2:6
    assert numpy.array_equal(out[0], data[0, 2:6, 2:6])


def test_crop_apply_mask_reduces_shape() -> None:
    data = numpy.ones((8, 8), dtype=bool)
    crop = CropStep(center=CropCenter(position_x_px=4, position_y_px=4), extent=ImageExtent(4, 4))
    assert crop.apply(data).shape == (4, 4)


# ---------- Binning ----------


def test_binning_apply_sums_blocks() -> None:
    data = numpy.ones((1, 4, 4), dtype=numpy.uint16)
    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)
    assert out.shape == (1, 2, 2)
    assert (out == 4).all()


def test_binning_apply_mask_logical_and() -> None:
    data = numpy.ones((4, 4), dtype=bool)
    data[0, 0] = False  # one True-cell of the (0,0) 2x2 block becomes False
    out = BinningStep(bin_size_x=2, bin_size_y=2).apply(data)
    assert out.shape == (2, 2)
    assert out[0, 0] == False  # logical AND of the block
    assert out[0, 1] == True
    assert out[1, 0] == True
    assert out[1, 1] == True


def test_binning_rejects_zero_bin_size() -> None:
    with pytest.raises(ValueError):
        BinningStep(bin_size_x=0, bin_size_y=1)


# ---------- Padding (B1) ----------


def test_padding_apply_3d_produces_correct_shape() -> None:
    """B1: pad_width must broadcast to (ndim, 2); flat tuples raise ValueError."""
    data = _zeros_patterns((2, 4, 4))
    out = PaddingStep(pad_x=1, pad_y=1).apply(data)
    assert out.shape == (2, 6, 6)
    assert (out == 0).all()


def test_padding_apply_mask_2d_produces_correct_shape() -> None:
    """B1 mirror: bad-pixels padding must not raise either."""
    data = numpy.ones((4, 4), dtype=bool)
    out = PaddingStep(pad_x=1, pad_y=1).apply(data)
    assert out.shape == (6, 6)
    # Edges are padded with False; interior preserved.
    assert out[0, 0] == False
    assert out[3, 3] == True


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


def test_apply_to_mask_full_pipeline() -> None:
    bad = numpy.zeros((8, 8), dtype=bool)
    bad[4, 4] = True
    out = _pipeline(
        CropStep(center=CropCenter(position_x_px=4, position_y_px=4), extent=ImageExtent(4, 4)),
        BinningStep(bin_size_x=2, bin_size_y=2),
        PaddingStep(pad_x=1, pad_y=1),
    ).apply_to_mask(bad)
    # 8x8 → crop to 4x4 (rows 2:6, cols 2:6, bad[4,4] inside) → bin 2x2 to 2x2 (logical AND so False) → pad to 4x4
    assert out.shape == (4, 4)


# ---------- Step extent / pixel-geometry ----------


def test_crop_apply_to_extent_returns_configured_extent() -> None:
    step = CropStep(center=CropCenter(position_x_px=4, position_y_px=4), extent=ImageExtent(4, 6))
    out = step.apply_to_extent(ImageExtent(64, 64))
    assert (out.width_px, out.height_px) == (4, 6)


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
    pipeline = _pipeline(
        CropStep(center=CropCenter(position_x_px=32, position_y_px=32), extent=ImageExtent(16, 16)),
        BinningStep(bin_size_x=2, bin_size_y=2),
        PaddingStep(pad_x=1, pad_y=1),
        TransposeStep(),
    )
    # 64x64 → crop 16x16 → bin 2x2 → 8x8 → pad → 10x10 → transpose → 10x10
    assert pipeline.compute_output_extent(ImageExtent(64, 64)) == ImageExtent(10, 10)


def test_pipeline_compute_output_pixel_geometry_composes_binning_and_transpose() -> None:
    pipeline = _pipeline(
        BinningStep(bin_size_x=2, bin_size_y=4),
        TransposeStep(),
    )
    out = pipeline.compute_output_pixel_geometry(PixelGeometry(width_m=1e-5, height_m=2e-5))
    # bin: width 2e-5, height 8e-5; transpose swaps → width 8e-5, height 2e-5
    assert out.width_m == 8e-5
    assert out.height_m == 2e-5


# ---------- Serialization ----------


def test_pipeline_serializes_and_round_trips() -> None:
    """Pydantic tagged-union round-trip so `ptychodus_store` can persist a pipeline."""
    original = _pipeline(
        FilterValuesStep(lower_bound=1, upper_bound=99),
        CropStep(center=CropCenter(position_x_px=4, position_y_px=4), extent=ImageExtent(4, 4)),
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
    extent = sizer.get_processed_image_extent()
    assert extent.width_px == 32 + 2 * 4
    assert extent.height_px == 32 + 2 * 4

    # Cross-check against the pipeline's actual output shape.
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 64, 64), dtype=numpy.uint16)
    )
    out_shape = sizer.get_prep_pipeline()(array).get_patterns().shape
    assert out_shape == (1, extent.height_px, extent.width_px)


def test_sizer_processed_extent_reflects_transpose(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """Regression: transpose must swap width/height in the processed extent + pixel geometry."""
    diff, det = settings
    det.width_px.set_value(64)
    det.height_px.set_value(32)
    det.pixel_width_m.set_value(1e-5)
    det.pixel_height_m.set_value(2e-5)
    diff.transpose.set_value(True)

    sizer = PatternSizer(det, diff)
    extent = sizer.get_processed_image_extent()
    assert (extent.width_px, extent.height_px) == (32, 64)

    geo = sizer.get_processed_pixel_geometry()
    assert (geo.width_m, geo.height_m) == (2e-5, 1e-5)

    # Cross-check: pipeline's actual output stack shape agrees with the sizer.
    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 32, 64), dtype=numpy.uint16)
    )
    out_shape = sizer.get_prep_pipeline()(array).get_patterns().shape
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

    pipeline = PatternSizer(det, diff).get_prep_pipeline()
    step = _filter_step(pipeline)
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

    pipeline = PatternSizer(det, diff).get_prep_pipeline()
    step = _filter_step(pipeline)
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

    pipeline = PatternSizer(det, diff).get_prep_pipeline()
    step = _filter_step(pipeline)
    assert step is not None
    assert step.lower_bound == 3
    assert step.upper_bound == 99


def test_sizer_no_filter_bounds(settings: tuple[DiffractionSettings, DetectorSettings]) -> None:
    """Both filter toggles off → no FilterValuesStep in the pipeline (avoid a no-op step)."""
    diff, det = settings
    diff.value_lower_bound_enabled.set_value(False)
    diff.value_upper_bound_enabled.set_value(False)

    pipeline = PatternSizer(det, diff).get_prep_pipeline()
    assert _filter_step(pipeline) is None


# ---------- Safe crop center ----------


def _crop_step(pipeline: DiffractionPrepPipeline) -> CropStep | None:
    for step in pipeline.steps:
        if isinstance(step, CropStep):
            return step
    return None


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
def test_sizer_safe_crop_center_matches_cropstep_bounds(
    settings: tuple[DiffractionSettings, DetectorSettings], user_center: int, expected: int
) -> None:
    """The clamped center must equal the CropStep radius-based bounds (see CropStep.apply)."""
    diff, det = settings
    det.width_px.set_value(8)
    det.height_px.set_value(8)
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(4)
    diff.crop_height_px.set_value(4)
    diff.crop_center_x_px.set_value(user_center)
    diff.crop_center_y_px.set_value(user_center)

    step = _crop_step(PatternSizer(det, diff).get_prep_pipeline())
    assert step is not None
    assert step.center.position_x_px == expected
    assert step.center.position_y_px == expected


def test_sizer_safe_crop_center_produces_in_bounds_slice(
    settings: tuple[DiffractionSettings, DetectorSettings],
) -> None:
    """Cross-check: a max-valid center must yield a slice fully inside the detector."""
    diff, det = settings
    det.width_px.set_value(8)
    det.height_px.set_value(8)
    diff.crop_enabled.set_value(True)
    diff.crop_width_px.set_value(4)
    diff.crop_height_px.set_value(4)
    diff.crop_center_x_px.set_value(6)  # max valid; radius = 2 → slice [4:8]
    diff.crop_center_y_px.set_value(6)

    array = SimpleDiffractionArray(
        'a', numpy.zeros(1, dtype=int), numpy.zeros((1, 8, 8), dtype=numpy.uint16)
    )
    out = PatternSizer(det, diff).get_prep_pipeline()(array).get_patterns()
    assert out.shape == (1, 4, 4)  # no truncation from an out-of-bounds slice


# ---------- Observer notifications ----------


class _CountingObserver:
    def __init__(self) -> None:
        self.n_updates = 0

    def _update(self, observable: object) -> None:
        self.n_updates += 1


@pytest.mark.parametrize(
    'attr',
    [
        'hflip',
        'vflip',
        'transpose',
        'value_lower_bound_enabled',
        'value_lower_bound',
        'value_upper_bound_enabled',
        'value_upper_bound',
    ],
)
def test_sizer_notifies_on_whole_image_parameter_change(
    settings: tuple[DiffractionSettings, DetectorSettings], attr: str
) -> None:
    """Whole-image settings (flips, transpose, filter bounds) must wake up sizer observers."""
    diff, det = settings
    sizer = PatternSizer(det, diff)
    observer = _CountingObserver()
    sizer.add_observer(observer)  # type: ignore[arg-type]

    parameter = getattr(diff, attr)
    current = parameter.get_value()
    parameter.set_value(current + 1 if isinstance(current, int) else not current)

    assert observer.n_updates >= 1
