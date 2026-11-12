"""Tests for the ptychodus native HDF5 diffraction reader and writer."""

from __future__ import annotations

from pathlib import Path

import numpy
import numpy.testing
import pytest

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.diffraction import (
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import save_diffraction_data
from ptychodus.plugins.h5_diffraction_file import H5DiffractionFileIO

_PIXEL_GEOMETRY = PixelGeometry(width_m=75e-6, height_m=55e-6)


def _make_dataset(
    *, pixel_geometry: PixelGeometry | None = _PIXEL_GEOMETRY
) -> SimpleDiffractionDataset:
    rng = numpy.random.default_rng(0)
    extent = ImageExtent(width_px=6, height_px=4)
    arrays = [
        SimpleDiffractionArray(
            label=f'array{array_index}',
            indexes=numpy.arange(3) + 10 * array_index,
            patterns=rng.integers(0, 1000, size=(3, 4, 6), dtype=numpy.uint16),
        )
        for array_index in range(2)
    ]
    bad_pixels = numpy.zeros(extent.get_shape(), dtype=numpy.bool)
    bad_pixels[1, 2] = True
    metadata = DiffractionMetadata(
        num_patterns_per_array=[3, 3],
        pattern_dtype=numpy.dtype(numpy.uint16),
        detector_extent=extent,
        detector_pixel_geometry=pixel_geometry,
    )
    return SimpleDiffractionDataset(
        metadata,
        DiffractionDatasetLayoutNode.create_root(),
        arrays,
        bad_pixels=bad_pixels,
        background=numpy.full(extent.get_shape(), 2.0),
        flat_field=numpy.full(extent.get_shape(), 0.9),
    )


def test_round_trip_concatenates_arrays_and_keeps_maps(tmp_path: Path) -> None:
    file_path = tmp_path / 'diffraction.h5'
    dataset = _make_dataset()
    file_io = H5DiffractionFileIO()

    file_io.write(file_path, dataset)
    loaded = file_io.read(file_path)

    assert len(loaded) == 1
    numpy.testing.assert_array_equal(
        loaded[0].get_indexes(), numpy.concatenate([a.get_indexes() for a in dataset])
    )
    numpy.testing.assert_array_equal(
        loaded[0].get_patterns(), numpy.concatenate([a.get_patterns() for a in dataset])
    )
    metadata = loaded.get_metadata()
    assert metadata.num_patterns_per_array == [6]
    assert metadata.detector_extent == ImageExtent(width_px=6, height_px=4)
    assert metadata.detector_pixel_geometry == _PIXEL_GEOMETRY
    assert metadata.file_path == file_path
    numpy.testing.assert_array_equal(loaded.get_bad_pixels(), dataset.get_bad_pixels())

    background = loaded.get_background()
    flat_field = loaded.get_flat_field()
    assert background is not None and flat_field is not None
    numpy.testing.assert_allclose(background, 2.0)
    numpy.testing.assert_allclose(flat_field, 0.9)


def test_writer_refuses_a_dataset_without_a_pixel_size(tmp_path: Path) -> None:
    file_path = tmp_path / 'diffraction.h5'

    with pytest.raises(ValueError, match='pixel size'):
        H5DiffractionFileIO().write(file_path, _make_dataset(pixel_geometry=None))

    assert not file_path.exists()


def test_reads_a_file_written_by_save_diffraction_data(tmp_path: Path) -> None:
    file_path = tmp_path / 'diffraction.h5'
    patterns = numpy.ones((5, 4, 6), dtype=numpy.float32)
    data = AssembledDiffractionData(
        numpy.arange(5),
        patterns,
        _PIXEL_GEOMETRY,
        numpy.zeros((4, 6), dtype=numpy.bool),
    )
    save_diffraction_data(file_path, data)

    loaded = H5DiffractionFileIO().read(file_path)

    assert loaded.get_metadata().pattern_dtype == numpy.dtype(numpy.float32)
    assert loaded.get_background() is None
    assert loaded.get_flat_field() is None
    numpy.testing.assert_array_equal(loaded[0].get_indexes(), numpy.arange(5))
