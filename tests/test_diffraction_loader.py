"""End-to-end tests for AssembledDiffractionDataset's load wiring.

These cover what dataset.py contributes on top of ptychodus.api.assemble: that
it derives the prep pipeline and masks from live settings, sizes the shared
buffer from the *processed* extent while metadata keeps the raw one, and places
each array at its metadata-derived offset. The pure preprocessing invariants
live in tests/test_assemble.py.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy

from ptychodus.api.diffraction import (
    DiffractionArray,
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.diffraction.dataset import AssembledDiffractionDataset
from ptychodus.model.diffraction.monitor import DiffractionTaskMonitor
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


class InlineTaskManager:
    """Runs queued tasks immediately, so loads complete before the call returns."""

    is_stopping = False
    background_queue_size = 0
    foreground_queue_size = 0

    def put_background_task(self, task: Callable[[], None]) -> None:
        task()

    def put_foreground_task(self, task: Callable[[], None]) -> None:
        task()


def make_dataset(
    *,
    detector_height: int = 40,
    detector_width: int = 60,
    crop_center_y: int = 20,
    crop_center_x: int = 30,
    crop_height: int = 12,
    crop_width: int = 16,
    arrays: Sequence[DiffractionArray] = (),
    num_patterns_per_array: Sequence[int] | None = None,
) -> AssembledDiffractionDataset:
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)

    diffraction_settings.crop_enabled.set_value(True)
    diffraction_settings.crop_center_y_px.set_value(crop_center_y)
    diffraction_settings.crop_center_x_px.set_value(crop_center_x)
    diffraction_settings.crop_height_px.set_value(crop_height)
    diffraction_settings.crop_width_px.set_value(crop_width)

    task_manager = InlineTaskManager()
    dataset = AssembledDiffractionDataset(
        diffraction_settings,
        PatternSizer(diffraction_settings),
        detector_settings,
        task_manager,  # type: ignore[arg-type]
        DiffractionTaskMonitor(task_manager),  # type: ignore[arg-type]
    )
    metadata = DiffractionMetadata(
        num_patterns_per_array=(
            [a.get_num_patterns() for a in arrays]
            if num_patterns_per_array is None
            else list(num_patterns_per_array)
        ),
        pattern_dtype=numpy.dtype(numpy.uint16),
        detector_extent=ImageExtent(width_px=detector_width, height_px=detector_height),
    )
    source = SimpleDiffractionDataset(metadata, DiffractionDatasetLayoutNode.create_root(), arrays)
    dataset.reload(source)
    return dataset


def _saturated_array(label: str, num_patterns: int, first_index: int = 0) -> SimpleDiffractionArray:
    """Patterns of 1s with a 65535 spike at raw (18, 30)."""
    patterns = numpy.ones((num_patterns, 40, 60), dtype=numpy.uint16)
    patterns[:, 18, 30] = 65535
    indexes = numpy.arange(first_index, first_index + num_patterns, dtype=numpy.intp)
    return SimpleDiffractionArray(label, indexes, patterns)


def test_load_applies_settings_pipeline_and_repairs_bad_pixels_before_crop() -> None:
    """Raw 40x60 cropped to 12x16; the masked spike is zeroed before the crop."""
    dataset = make_dataset(arrays=[_saturated_array('test', 3)])
    raw_bad = numpy.zeros((40, 60), dtype=bool)
    raw_bad[18, 30] = True  # crop [y=14..26, x=22..38] puts this at (4, 8)
    dataset.set_bad_pixels(raw_bad)

    dataset.load_all_arrays(process_patterns=True, block=True)

    array = dataset[0]
    bad_pixels = dataset.get_assembled_data().get_bad_pixels()
    assert array.get_patterns().shape == (3, 12, 16)
    assert bad_pixels.shape == (12, 16)
    assert bad_pixels[4, 8]
    assert int(array.get_pattern(0)[4, 8]) == 0
    assert int(array.get_pattern(0)[0, 0]) == 1


def test_load_all_arrays_preserves_raw_metadata_extent() -> None:
    """Metadata keeps the raw detector extent; the buffer holds the processed mask."""
    dataset = make_dataset()

    dataset.load_all_arrays(process_patterns=True, block=True)

    extent = dataset.get_metadata().detector_extent
    assert extent.width_px == 60
    assert extent.height_px == 40
    assert dataset.get_assembled_data().get_bad_pixels().shape == (12, 16)


def test_load_without_processing_keeps_raw_shapes() -> None:
    """process_patterns=False skips the pipeline but still zeroes bad pixels."""
    patterns = numpy.full((2, 40, 60), 7, dtype=numpy.uint16)
    patterns[:, 5, 5] = 999
    array = SimpleDiffractionArray('raw', numpy.arange(2, dtype=numpy.intp), patterns)
    dataset = make_dataset(arrays=[array])
    raw_bad = numpy.zeros((40, 60), dtype=bool)
    raw_bad[5, 5] = True
    dataset.set_bad_pixels(raw_bad)

    dataset.load_all_arrays(process_patterns=False, block=True)

    loaded = dataset[0]
    assert loaded.get_patterns().shape == (2, 40, 60)
    assert dataset.get_assembled_data().get_bad_pixels().shape == (40, 60)
    assert int(loaded.get_pattern(0)[5, 5]) == 0
    assert int(loaded.get_pattern(0)[0, 0]) == 7


def test_arrays_land_at_their_metadata_offsets_in_order() -> None:
    dataset = make_dataset(
        arrays=[_saturated_array('a', 2, 0), _saturated_array('b', 3, 2)],
    )

    dataset.load_all_arrays(process_patterns=True, block=True)

    assert [a.get_label() for a in dataset] == ['a', 'b']
    numpy.testing.assert_array_equal(dataset.get_assembled_data().get_indexes(), [0, 1, 2, 3, 4])


def test_append_array_after_a_load_continues_past_the_loaded_arrays() -> None:
    """Regression: the appended array must not overwrite array 0 at offset 0."""
    dataset = make_dataset(
        arrays=[_saturated_array('a', 2, 0)],
        num_patterns_per_array=[2, 2],
    )
    dataset.load_all_arrays(process_patterns=True, block=True)

    dataset.append_array(_saturated_array('b', 2, 10), process_patterns=True)

    assert [a.get_label() for a in dataset] == ['a', 'b']
    numpy.testing.assert_array_equal(dataset.get_assembled_data().get_indexes(), [0, 1, 10, 11])


def test_processed_pixel_geometry_folds_binning_into_the_assembled_data() -> None:
    dataset = make_dataset(arrays=[_saturated_array('a', 2, 0)])
    dataset.set_pixel_geometry_override(PixelGeometry(width_m=1e-4, height_m=2e-4))

    dataset.load_all_arrays(process_patterns=True, block=True)

    # Crop alone leaves the pixel size untouched.
    assert dataset.get_assembled_data().get_pixel_geometry() == PixelGeometry(1e-4, 2e-4)
    assert dataset.get_raw_pixel_geometry() == PixelGeometry(1e-4, 2e-4)
