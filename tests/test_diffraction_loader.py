"""Regression tests for the per-array LoadArray pipeline.

Covers the invariant that when pattern processing is enabled, the bad-pixel
mask handed to the per-array AssembledDiffractionData matches the processed
patterns' shape, and that bad-pixel repair (zeroing) is applied to raw
patterns before crop/bin/pad/flip/transpose runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy

from ptychodus.api.diffraction import (
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.tree import SimpleTreeNode
from ptychodus.model.diffraction.dataset import AssembledDiffractionDataset
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


def _make_dataset(
    detector_height: int,
    detector_width: int,
    crop_center_y: int,
    crop_center_x: int,
    crop_height: int,
    crop_width: int,
) -> AssembledDiffractionDataset:
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)

    diffraction_settings.crop_enabled.set_value(True)
    diffraction_settings.crop_center_y_px.set_value(crop_center_y)
    diffraction_settings.crop_center_x_px.set_value(crop_center_x)
    diffraction_settings.crop_height_px.set_value(crop_height)
    diffraction_settings.crop_width_px.set_value(crop_width)

    sizer = PatternSizer(diffraction_settings)
    task_manager = MagicMock()
    task_monitor = MagicMock()

    dataset = AssembledDiffractionDataset(
        diffraction_settings,
        sizer,
        detector_settings,
        task_manager,
        task_monitor,
    )
    # Give the dataset a metadata source with the intended detector extent so the
    # per-dataset extent flows through reload() -> the sizer's pipeline builder.
    metadata = DiffractionMetadata(
        num_patterns_per_array=[0],
        pattern_dtype=numpy.dtype(numpy.uint16),
        detector_extent=ImageExtent(width_px=detector_width, height_px=detector_height),
    )
    contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
    source = SimpleDiffractionDataset(metadata, contents_tree, [])
    dataset.reload(source)
    return dataset


def test_load_array_matches_shapes_and_repairs_bad_pixels_before_crop() -> None:
    """Reproduces the fly001.ini warning: raw 40x60 → crop to 12x16.

    Without the fix, LoadArray hands AssembledDiffractionData mismatched shapes
    (patterns (12,16) vs bad_pixels (40,60)) and ValueError propagates. With
    the fix, shapes match, the raw bad pixel appears at its cropped location,
    and its raw huge value has been zeroed rather than surviving into the crop.
    """
    dataset = _make_dataset(
        detector_height=40,
        detector_width=60,
        crop_center_y=20,
        crop_center_x=30,
        crop_height=12,
        crop_width=16,
    )

    # Bad pixel at raw (18, 30). Crop [y=14..26, x=22..38] → cropped (4, 8).
    raw_bad = numpy.zeros((40, 60), dtype=bool)
    raw_bad[18, 30] = True
    dataset.set_bad_pixels(raw_bad)

    raw_patterns = numpy.ones((3, 40, 60), dtype=numpy.uint16)
    raw_patterns[:, 18, 30] = 65535  # saturation that must not leak into the crop
    array = SimpleDiffractionArray('test', numpy.arange(3, dtype=numpy.intp), raw_patterns)

    captured: dict[str, object] = {}

    def stub_assemble_array(array_index, label, data):  # type: ignore[no-untyped-def]
        captured['data'] = data

    dataset.assemble_array = stub_assemble_array  # type: ignore[method-assign]

    task = dataset.create_array_loader(array, process_patterns=True)
    task()

    data = captured['data']
    assert data.get_patterns_shape() == (3, 12, 16)  # type: ignore[attr-defined]
    assert data.get_bad_pixels().shape == (12, 16)  # type: ignore[attr-defined]
    assert data.get_bad_pixels()[4, 8]  # type: ignore[attr-defined]
    # Repair happened before crop: the saturated raw value was zeroed.
    assert int(data.get_pattern(0)[4, 8]) == 0  # type: ignore[attr-defined]
    # Sanity: an unmasked pixel elsewhere in the crop still holds its raw value.
    assert int(data.get_pattern(0)[0, 0]) == 1  # type: ignore[attr-defined]


def test_load_all_arrays_preserves_raw_metadata_extent() -> None:
    """load_all_arrays(process_patterns=True) must not conflate raw detector geometry
    with the processed bad-pixel mask shape.

    Regression for the fly001.ini failure: the old code rebuilt the internal
    SimpleDiffractionDataset with raw metadata but the *processed* bad_pixels,
    tripping SimpleDiffractionDataset's shape-mismatch check. The processed
    mask belongs on self._data (AssembledDiffractionData), not on the metadata
    holder.
    """
    dataset = _make_dataset(
        detector_height=40,
        detector_width=60,
        crop_center_y=20,
        crop_center_x=30,
        crop_height=12,
        crop_width=16,
    )

    dataset.load_all_arrays(process_patterns=True, block=True)

    # Metadata still reports the raw detector extent.
    extent = dataset.get_metadata().detector_extent
    assert extent.width_px == 60
    assert extent.height_px == 40
    # The assembled data holds the processed bad-pixel mask matching the crop output.
    assert dataset.get_assembled_data().get_bad_pixels().shape == (12, 16)


def test_load_array_without_processing_keeps_raw_shapes() -> None:
    """process_patterns=False should skip the processor and keep raw shapes intact."""
    dataset = _make_dataset(
        detector_height=40,
        detector_width=60,
        crop_center_y=20,
        crop_center_x=30,
        crop_height=12,
        crop_width=16,
    )

    raw_bad = numpy.zeros((40, 60), dtype=bool)
    raw_bad[5, 5] = True
    dataset.set_bad_pixels(raw_bad)

    raw_patterns = numpy.full((2, 40, 60), 7, dtype=numpy.uint16)
    raw_patterns[:, 5, 5] = 999
    array = SimpleDiffractionArray('raw', numpy.arange(2, dtype=numpy.intp), raw_patterns)

    captured: dict[str, object] = {}

    def stub_assemble_array(array_index, label, data):  # type: ignore[no-untyped-def]
        captured['data'] = data

    dataset.assemble_array = stub_assemble_array  # type: ignore[method-assign]

    task = dataset.create_array_loader(array, process_patterns=False)
    task()

    data = captured['data']
    assert data.get_patterns_shape() == (2, 40, 60)  # type: ignore[attr-defined]
    assert data.get_bad_pixels().shape == (40, 60)  # type: ignore[attr-defined]
    # Repair still runs because the raw mask is meaningful even without geometric processing.
    assert int(data.get_pattern(0)[5, 5]) == 0  # type: ignore[attr-defined]
    assert int(data.get_pattern(0)[0, 0]) == 7  # type: ignore[attr-defined]
