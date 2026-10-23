"""Tests for the measured detector background and flat field.

Covers the corrections themselves, how they travel through assembly, persistence in
the assembled HDF5 file, the reconstruction input, the pixel-map readers, and the
diffraction model that builds corrections from its settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy
import pytest

from ptychodus.api.assemble import (
    AssembledDiffractionData,
    DetectorCorrections,
    allocate_assembled_data,
    assemble_dataset,
    compute_assembled_dtype,
    compute_dataset_total_counts,
)
from ptychodus.api.diffraction import (
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    DiffractionPatternDType,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.io import DiffractionFileKeys, load_diffraction_data, save_diffraction_data
from ptychodus.api.preprocess.diffraction import (
    BinningStep,
    DiffractionPrepPipeline,
    FilterValuesStep,
    PaddingStep,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.cli import add_detector_map_arguments, read_detector_corrections
from ptychodus.model.diffraction.dataset import AssembledDiffractionDataset
from ptychodus.model.diffraction.monitor import DiffractionTaskMonitor
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.plugins.pixel_maps import (
    NPYFlatFieldCorrectionFileReader,
    NPYPixelMapFileReader,
    register_plugins,
)

from diffraction_dataset_fixtures import GEOMETRY, make_dataset
from test_diffraction_loader import InlineTaskManager

FRAME_SHAPE = (4, 4)


def _uint16_array(values: numpy.ndarray, num_patterns: int = 2) -> SimpleDiffractionArray:
    patterns = numpy.broadcast_to(values, (num_patterns, *values.shape)).astype(numpy.uint16)
    return SimpleDiffractionArray('a', numpy.arange(num_patterns, dtype=numpy.intp), patterns)


# ---------- DetectorCorrections ----------


def test_null_corrections_change_nothing() -> None:
    patterns = numpy.ones((1, 2, 2), dtype=numpy.uint16)
    corrections = DetectorCorrections.create_null()

    assert corrections.apply(patterns) is patterns
    assert corrections.apply_to_dtype(numpy.dtype(numpy.uint16)) == numpy.uint16


def test_maps_without_their_flags_are_carried_not_applied() -> None:
    corrections = DetectorCorrections(background=numpy.ones((2, 2)), flat_field=numpy.ones((2, 2)))
    patterns = numpy.full((1, 2, 2), 5, dtype=numpy.uint16)

    assert corrections.apply(patterns) is patterns
    assert not corrections.applies_background
    assert not corrections.applies_flat_field


def test_correction_subtracts_divides_and_clips() -> None:
    corrections = DetectorCorrections(
        background=numpy.array([[1.0, 10.0]]),
        flat_field=numpy.array([[0.5, 2.0]]),
        subtract_background=True,
        divide_flat_field=True,
    )
    patterns = numpy.array([[[5, 4]]], dtype=numpy.uint16)

    corrected = corrections.apply(patterns)

    assert corrected.dtype == numpy.float32
    numpy.testing.assert_allclose(corrected, [[[(5 - 1) / 0.5, 0.0]]])


def test_double_precision_patterns_stay_double() -> None:
    corrections = DetectorCorrections(background=numpy.zeros((1, 1)), subtract_background=True)

    assert corrections.apply_to_dtype(numpy.dtype(numpy.float64)) == numpy.float64
    assert corrections.apply_to_dtype(numpy.dtype(numpy.int32)) == numpy.float32


def test_low_response_marks_pixels_bad_only_when_dividing() -> None:
    flat_field = numpy.array([[1.0, 1.0e-6, 0.0]])
    clean = numpy.zeros((1, 3), dtype=bool)

    dividing = DetectorCorrections(flat_field=flat_field, divide_flat_field=True)
    carrying = DetectorCorrections(flat_field=flat_field)

    numpy.testing.assert_array_equal(dividing.apply_to_bad_pixels(clean), [[False, True, True]])
    numpy.testing.assert_array_equal(carrying.apply_to_bad_pixels(clean), [[False, False, False]])


def test_non_finite_map_entries_mark_pixels_bad() -> None:
    background = numpy.array([[0.0, numpy.nan, 0.0]])
    flat_field = numpy.array([[1.0, 1.0, numpy.inf]])
    corrections = DetectorCorrections(background=background, flat_field=flat_field)

    widened = corrections.apply_to_bad_pixels(numpy.zeros((1, 3), dtype=bool))

    numpy.testing.assert_array_equal(widened, [[False, True, True]])


def test_through_pipeline_sums_the_background_and_averages_the_flat_field() -> None:
    corrections = DetectorCorrections(
        background=numpy.ones(FRAME_SHAPE), flat_field=numpy.arange(16.0).reshape(FRAME_SHAPE)
    )
    pipeline = DiffractionPrepPipeline(
        steps=(BinningStep(bin_size_x=2, bin_size_y=2), PaddingStep(pad_x=1, pad_y=0))
    )

    processed = corrections.through_pipeline(pipeline)

    assert processed.background is not None and processed.flat_field is not None
    numpy.testing.assert_array_equal(processed.background, [[0, 4, 4, 0], [0, 4, 4, 0]])
    numpy.testing.assert_allclose(processed.flat_field, [[1, 2.5, 4.5, 1], [1, 10.5, 12.5, 1]])


# ---------- assembly ----------


def test_unapplied_maps_are_carried_on_the_processed_grid() -> None:
    background = numpy.full(FRAME_SHAPE, 2.0)
    flat_field = numpy.full(FRAME_SHAPE, 0.5)
    dataset = make_dataset([_uint16_array(numpy.full(FRAME_SHAPE, 9))], FRAME_SHAPE)
    pipeline = DiffractionPrepPipeline(steps=(BinningStep(bin_size_x=2, bin_size_y=2),))
    corrections = DetectorCorrections(background=background, flat_field=flat_field)

    data = assemble_dataset(dataset, pipeline, corrections=corrections, max_workers=1)

    # Nothing applied, so the patterns stay integer (binning only widens them).
    assert numpy.issubdtype(data.get_patterns_dtype(), numpy.integer)
    numpy.testing.assert_array_equal(data.get_patterns(), 36)
    numpy.testing.assert_array_equal(data.get_background(), 8.0)
    numpy.testing.assert_array_equal(data.get_flat_field(), 0.5)


def test_applied_maps_correct_the_patterns_and_are_dropped() -> None:
    dataset = make_dataset([_uint16_array(numpy.full(FRAME_SHAPE, 9))], FRAME_SHAPE)
    pipeline = DiffractionPrepPipeline(steps=(BinningStep(bin_size_x=2, bin_size_y=2),))
    corrections = DetectorCorrections(
        background=numpy.full(FRAME_SHAPE, 1.0),
        flat_field=numpy.full(FRAME_SHAPE, 2.0),
        subtract_background=True,
        divide_flat_field=True,
    )

    data = assemble_dataset(dataset, pipeline, corrections=corrections, max_workers=1)

    assert data.get_patterns_dtype() == numpy.float32
    # Binned counts 36, binned background 4, mean response 2.
    numpy.testing.assert_allclose(data.get_patterns(), (36 - 4) / 2)
    assert data.get_background() is None
    assert data.get_flat_field() is None


def test_value_filters_act_on_measured_counts_not_corrected_ones() -> None:
    """An invalid-value marker is caught before the background shifts it."""
    values = numpy.full(FRAME_SHAPE, 10)
    values[1, 1] = 65535
    dataset = make_dataset([_uint16_array(values)], FRAME_SHAPE)
    pipeline = DiffractionPrepPipeline(steps=(FilterValuesStep(lower_bound=0, upper_bound=65535),))
    corrections = DetectorCorrections(
        background=numpy.full(FRAME_SHAPE, 4.0), subtract_background=True
    )

    data = assemble_dataset(dataset, pipeline, corrections=corrections, max_workers=1)

    assert data.get_patterns()[0, 1, 1] == 0.0
    assert data.get_patterns()[0, 0, 0] == pytest.approx(6.0)


def test_unusable_flat_field_pixels_reach_the_stored_mask() -> None:
    flat_field = numpy.ones(FRAME_SHAPE)
    flat_field[2, 3] = 0.0
    dataset = make_dataset([_uint16_array(numpy.full(FRAME_SHAPE, 3))], FRAME_SHAPE)
    corrections = DetectorCorrections(flat_field=flat_field, divide_flat_field=True)

    data = assemble_dataset(dataset, corrections=corrections, max_workers=1)

    assert data.get_bad_pixels()[2, 3]
    assert data.get_patterns()[0, 2, 3] == 0.0
    assert numpy.isfinite(data.get_patterns()).all()


def test_the_dataset_maps_are_carried_by_default() -> None:
    background = numpy.full(FRAME_SHAPE, 1.5)
    source = SimpleDiffractionDataset(
        DiffractionMetadata(
            num_patterns_per_array=[1],
            pattern_dtype=numpy.dtype(numpy.uint16),
            detector_extent=ImageExtent(width_px=4, height_px=4),
            detector_pixel_geometry=GEOMETRY,
        ),
        DiffractionDatasetLayoutNode.create_root(),
        [_uint16_array(numpy.full(FRAME_SHAPE, 2), num_patterns=1)],
        background=background,
    )

    data = assemble_dataset(source, max_workers=1)

    numpy.testing.assert_array_equal(data.get_background(), background)


def test_total_counts_are_measured_after_correction() -> None:
    dataset = make_dataset([_uint16_array(numpy.full(FRAME_SHAPE, 5))], FRAME_SHAPE)
    corrections = DetectorCorrections(
        background=numpy.full(FRAME_SHAPE, 1.0), subtract_background=True
    )

    counts = compute_dataset_total_counts(dataset, corrections=corrections, max_workers=1)

    numpy.testing.assert_allclose(counts.total_counts, [4.0 * 16] * 2)


def test_assembled_dtype_folds_the_pipeline_before_the_correction() -> None:
    dataset = make_dataset([_uint16_array(numpy.ones(FRAME_SHAPE))], FRAME_SHAPE)
    pipeline = DiffractionPrepPipeline(steps=(BinningStep(bin_size_x=2, bin_size_y=2),))
    corrections = DetectorCorrections(flat_field=numpy.ones(FRAME_SHAPE), divide_flat_field=True)

    assert compute_assembled_dtype(dataset, pipeline) == numpy.uint32
    assert compute_assembled_dtype(dataset, pipeline, corrections=corrections) == numpy.float32


def test_a_map_of_the_wrong_shape_is_rejected() -> None:
    dataset = make_dataset([_uint16_array(numpy.ones(FRAME_SHAPE))], FRAME_SHAPE)

    with pytest.raises(ValueError, match='Background shape'):
        allocate_assembled_data(
            dataset, corrections=DetectorCorrections(background=numpy.zeros((2, 2)))
        )


# ---------- AssembledDiffractionData ----------


def _bare_data(**kwargs: numpy.ndarray) -> AssembledDiffractionData:
    return AssembledDiffractionData(
        indexes=numpy.arange(2),
        patterns=numpy.zeros((2, 2, 2)),
        pixel_geometry=GEOMETRY,
        bad_pixels=numpy.zeros((2, 2), dtype=bool),
        **kwargs,
    )


def test_assembled_maps_must_match_the_pattern_grid() -> None:
    with pytest.raises(ValueError, match='Flat field shape'):
        _bare_data(flat_field=numpy.ones((3, 3)))


def test_assembled_maps_must_be_finite() -> None:
    with pytest.raises(ValueError, match='finite'):
        _bare_data(background=numpy.array([[0.0, numpy.nan], [0.0, 0.0]]))


def test_assembled_maps_are_read_only_and_counted() -> None:
    data = _bare_data(background=numpy.ones((2, 2)), flat_field=numpy.ones((2, 2)))
    background = data.get_background()

    assert background is not None and not background.flags.writeable
    assert data.nbytes == _bare_data().nbytes + 2 * numpy.ones((2, 2)).nbytes


def test_views_from_assemble_carry_the_buffer_maps() -> None:
    background = numpy.ones((2, 2))
    data = _bare_data(background=background)
    block = _bare_data()

    view = data.assemble(block, 0)

    numpy.testing.assert_array_equal(view.get_background(), background)


# ---------- persistence and reconstruction input ----------


def test_assembled_file_round_trips_the_maps(tmp_path: Path) -> None:
    path = tmp_path / 'diffraction.h5'
    data = _bare_data(background=numpy.full((2, 2), 3.0), flat_field=numpy.full((2, 2), 0.9))

    save_diffraction_data(path, data)
    loaded = load_diffraction_data(path)

    numpy.testing.assert_array_equal(loaded.get_background(), 3.0)
    numpy.testing.assert_array_equal(loaded.get_flat_field(), 0.9)


def test_assembled_file_without_maps_loads_without_them(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / 'diffraction.h5'
    save_diffraction_data(path, _bare_data())

    with h5py.File(path, 'r') as h5_file:
        assert DiffractionFileKeys.BACKGROUND not in h5_file
        assert DiffractionFileKeys.FLAT_FIELD not in h5_file

    loaded = load_diffraction_data(path)
    assert loaded.get_background() is None
    assert loaded.get_flat_field() is None


# ---------- pixel-map readers ----------


def test_npy_reader_returns_a_float_map(tmp_path: Path) -> None:
    path = tmp_path / 'map.npy'
    numpy.save(path, numpy.array([[1, 2], [3, 4]], dtype=numpy.uint16))

    pixel_map = NPYPixelMapFileReader('NumPy Background Files').read(path)

    assert pixel_map.dtype == numpy.float64
    numpy.testing.assert_array_equal(pixel_map, [[1, 2], [3, 4]])


def test_npy_reader_rejects_a_stack(tmp_path: Path) -> None:
    path = tmp_path / 'stack.npy'
    numpy.save(path, numpy.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match='2-D'):
        NPYPixelMapFileReader('NumPy Background Files').read(path)


def test_correction_factors_read_as_their_reciprocal_response(tmp_path: Path) -> None:
    path = tmp_path / 'factors.npy'
    numpy.save(path, numpy.array([[2.0, 0.5, 0.0, -1.0]]))

    response = NPYFlatFieldCorrectionFileReader().read(path)

    numpy.testing.assert_array_equal(response, [[0.5, 2.0, 0.0, 0.0]])


# ---------- diffraction model ----------


def _model_dataset(
    *,
    patterns: numpy.ndarray,
    memmap_directory: Path | None = None,
) -> tuple[AssembledDiffractionDataset, DetectorSettings]:
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)
    diffraction_settings.crop_enabled.set_value(False)

    if memmap_directory is not None:
        diffraction_settings.memmap_enabled.set_value(True)
        diffraction_settings.scratch_directory.set_value(memmap_directory)

    task_manager = InlineTaskManager()
    dataset = AssembledDiffractionDataset(
        diffraction_settings, detector_settings, task_manager, DiffractionTaskMonitor(task_manager)
    )
    height, width = patterns.shape[1:]
    metadata = DiffractionMetadata(
        num_patterns_per_array=[patterns.shape[0]],
        pattern_dtype=cast(DiffractionPatternDType, patterns.dtype),
        detector_extent=ImageExtent(width_px=width, height_px=height),
        detector_pixel_geometry=GEOMETRY,
    )
    array = SimpleDiffractionArray('a', numpy.arange(patterns.shape[0]), patterns)
    dataset.reload(
        SimpleDiffractionDataset(metadata, DiffractionDatasetLayoutNode.create_root(), [array])
    )
    return dataset, detector_settings


def test_model_applies_the_corrections_its_settings_enable(tmp_path: Path) -> None:
    patterns = numpy.full((2, 4, 4), 7, dtype=numpy.uint16)
    dataset, settings = _model_dataset(patterns=patterns, memmap_directory=tmp_path)
    settings.subtract_background.set_value(True)
    dataset.set_background(numpy.full((4, 4), 3.0))

    dataset.load_all_arrays(process_patterns=True, block=True)

    assembled = dataset.get_assembled_data()
    # The scratch file is sized for the promoted dtype, not the measured one.
    assert assembled.get_patterns_dtype() == numpy.float32
    numpy.testing.assert_allclose(assembled.get_patterns(), 4.0)
    assert assembled.get_background() is None


def test_model_carries_maps_its_settings_do_not_apply() -> None:
    patterns = numpy.full((1, 4, 4), 7, dtype=numpy.uint16)
    dataset, _ = _model_dataset(patterns=patterns)
    dataset.set_flat_field(numpy.full((4, 4), 0.8))

    dataset.load_all_arrays(process_patterns=True, block=True)

    flat_field = dataset.get_assembled_data().get_flat_field()
    assert flat_field is not None
    numpy.testing.assert_allclose(flat_field, 0.8)


def test_model_maps_follow_the_dataset_lifecycle() -> None:
    patterns = numpy.ones((1, 4, 4), dtype=numpy.uint16)
    dataset, _ = _model_dataset(patterns=patterns)

    with pytest.raises(ValueError, match='Background shape'):
        dataset.set_background(numpy.zeros((2, 2)))

    dataset.set_background(numpy.zeros((4, 4)))
    dataset.set_flat_field(numpy.ones((4, 4)))
    assert dataset.get_background() is not None

    dataset.reset_background()
    assert dataset.get_background() is None

    dataset.clear()
    assert dataset.get_flat_field() is None


def test_model_reimports_the_carried_maps(tmp_path: Path) -> None:
    patterns = numpy.ones((1, 4, 4), dtype=numpy.uint16)
    dataset, _ = _model_dataset(patterns=patterns)
    dataset.set_background(numpy.full((4, 4), 2.0))
    dataset.load_all_arrays(process_patterns=True, block=True)
    path = tmp_path / 'exported.h5'

    dataset.export_assembled_patterns(path)
    reimported, _ = _model_dataset(patterns=patterns)
    reimported.import_assembled_patterns(path)

    numpy.testing.assert_array_equal(reimported.get_background(), 2.0)
    numpy.testing.assert_array_equal(reimported.get_assembled_data().get_background(), 2.0)


# ---------- command-line options ----------


def test_command_line_maps_override_the_dataset_maps(tmp_path: Path) -> None:
    import argparse

    registry = PluginRegistry()
    register_plugins(registry)
    parser = argparse.ArgumentParser()
    add_detector_map_arguments(parser)
    flat_path = tmp_path / 'flat.npy'
    numpy.save(flat_path, numpy.full(FRAME_SHAPE, 0.25))
    args = parser.parse_args(['--flat-field-file', str(flat_path), '--divide-flat-field'])
    background = numpy.full(FRAME_SHAPE, 1.0)
    source = SimpleDiffractionDataset(
        DiffractionMetadata(
            num_patterns_per_array=[0],
            pattern_dtype=numpy.dtype(numpy.uint16),
            detector_extent=ImageExtent(width_px=4, height_px=4),
        ),
        DiffractionDatasetLayoutNode.create_root(),
        [],
        background=background,
    )

    corrections = read_detector_corrections(registry, args, source)

    assert corrections.background is background
    assert not corrections.subtract_background
    numpy.testing.assert_array_equal(corrections.flat_field, 0.25)
    assert corrections.divide_flat_field


# ---------- reconstruction input ----------


def test_reconstruct_input_carries_the_unapplied_maps() -> None:
    from ptychodus.api.reconstruct import prepare_reconstruct_input
    from test_io import _make_product

    product = _make_product(num_positions=3)
    data = AssembledDiffractionData(
        indexes=product.probe_positions.get_indexes().copy(),
        patterns=numpy.ones((3, 2, 2)),
        pixel_geometry=GEOMETRY,
        bad_pixels=numpy.zeros((2, 2), dtype=bool),
        background=numpy.full((2, 2), 6.0),
        flat_field=numpy.full((2, 2), 0.7),
    )

    reconstruct_input = prepare_reconstruct_input(data, product)

    assert reconstruct_input.background is not None and reconstruct_input.flat_field is not None
    numpy.testing.assert_array_equal(reconstruct_input.background, 6.0)
    numpy.testing.assert_array_equal(reconstruct_input.flat_field, 0.7)
