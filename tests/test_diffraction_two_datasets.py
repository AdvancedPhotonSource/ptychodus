"""Two-dataset integration test for the diffraction model.

Exercises the DiffractionDatasetRepository together with real
AssembledDiffractionDataset instances (built via the repository's factory)
to verify per-dataset bad-pixels ownership and stable index-based routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy
import pytest

from ptychodus.api.diffraction import DiffractionMetadata, SimpleDiffractionDataset
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import AssembledDiffractionData
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.tree import SimpleTreeNode
from ptychodus.model.diffraction.dataset import (
    AssembledDiffractionArray,
    AssembledDiffractionDataset,
)
from ptychodus.model.diffraction.repository import DiffractionDatasetRepository
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.diffraction.sizer import PatternSizer


def _make_repository() -> DiffractionDatasetRepository:
    registry = SettingsRegistry()
    detector_settings = DetectorSettings(registry)
    diffraction_settings = DiffractionSettings(registry)
    sizer = PatternSizer(detector_settings, diffraction_settings)
    task_manager = MagicMock()
    task_monitor = MagicMock()

    def _factory(name: str) -> AssembledDiffractionDataset:
        return AssembledDiffractionDataset(
            diffraction_settings,
            sizer,
            detector_settings,
            task_manager,
            task_monitor,
            name=name,
        )

    return DiffractionDatasetRepository(factory=_factory)


def _reload_with_extent(dataset: AssembledDiffractionDataset, extent: ImageExtent) -> None:
    metadata = DiffractionMetadata(
        num_patterns_per_array=[0],
        pattern_dtype=numpy.dtype(numpy.uint16),
        detector_extent=extent,
    )
    contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
    source = SimpleDiffractionDataset(metadata, contents_tree, [])
    dataset.reload(source)


def test_two_datasets_end_up_at_stable_indexes() -> None:
    repo = _make_repository()

    a = repo.create_dataset('scan_a')
    index_a = repo.insert_dataset(a)

    b = repo.create_dataset('scan_b')
    index_b = repo.insert_dataset(b)

    assert index_a == 0
    assert index_b == 1
    assert repo[0].get_name() == 'scan_a'
    assert repo[1].get_name() == 'scan_b'


def test_bad_pixels_are_per_dataset() -> None:
    repo = _make_repository()

    a = repo.create_dataset('scan_a')
    repo.insert_dataset(a)
    _reload_with_extent(a, ImageExtent(width_px=16, height_px=16))
    b = repo.create_dataset('scan_b')
    repo.insert_dataset(b)
    _reload_with_extent(b, ImageExtent(width_px=8, height_px=8))

    # Each dataset's default mask is sized to *its own* detector extent — regression
    # against the earlier singleton-source behavior where both would have shared shape.
    assert repo[0].get_bad_pixels().shape == (16, 16)
    assert repo[1].get_bad_pixels().shape == (8, 8)
    assert not repo[0].get_bad_pixels().any()
    assert not repo[1].get_bad_pixels().any()

    # Set a distinctive mask on dataset A only.
    custom_mask_a = numpy.zeros_like(repo[0].get_bad_pixels())
    custom_mask_a[0, 0] = True
    repo[0].set_bad_pixels(custom_mask_a)

    # A picks it up; B is untouched.
    assert repo[0].get_bad_pixels()[0, 0]
    assert not repo[1].get_bad_pixels()[0, 0]

    # And they are not aliased.
    assert repo[0].get_bad_pixels() is not repo[1].get_bad_pixels()


def test_removing_first_dataset_shifts_indexes() -> None:
    repo = _make_repository()

    repo.insert_dataset(repo.create_dataset('scan_a'))
    b = repo.create_dataset('scan_b')
    repo.insert_dataset(b)

    repo.remove_dataset(0)

    assert len(repo) == 1
    assert repo[0] is b
    assert repo[0].get_name() == 'scan_b'


def test_reset_bad_pixels_restores_default() -> None:
    repo = _make_repository()
    a = repo.create_dataset('scan_a')
    repo.insert_dataset(a)
    _reload_with_extent(a, ImageExtent(width_px=16, height_px=16))

    mask = numpy.zeros_like(a.get_bad_pixels())
    mask[1, 1] = True
    a.set_bad_pixels(mask)
    assert a.get_bad_pixels()[1, 1]

    a.reset_bad_pixels()
    assert a.get_bad_pixels().shape == (16, 16)
    assert not a.get_bad_pixels().any()


def test_set_bad_pixels_rejects_shape_mismatch_after_reload() -> None:
    repo = _make_repository()
    a = repo.create_dataset('scan_a')
    repo.insert_dataset(a)
    _reload_with_extent(a, ImageExtent(width_px=16, height_px=16))

    wrong_shape = numpy.zeros((8, 8), dtype=numpy.bool_)
    with pytest.raises(ValueError, match='does not match loaded detector extent'):
        a.set_bad_pixels(wrong_shape)

    # The original default mask is untouched.
    assert a.get_bad_pixels().shape == (16, 16)


def test_set_bad_pixels_permissive_before_reload() -> None:
    """Streaming pre-load path: without a reloaded dataset, any 2-D mask is accepted."""
    repo = _make_repository()
    a = repo.create_dataset('scan_a')
    repo.insert_dataset(a)

    mask = numpy.zeros((32, 64), dtype=numpy.bool_)
    a.set_bad_pixels(mask)
    assert a.get_bad_pixels().shape == (32, 64)


def test_simple_diffraction_dataset_rejects_bad_pixels_shape_mismatch() -> None:
    metadata = DiffractionMetadata(
        num_patterns_per_array=[0],
        pattern_dtype=numpy.dtype(numpy.uint16),
        detector_extent=ImageExtent(width_px=16, height_px=16),
    )
    contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
    wrong_shape = numpy.zeros((8, 8), dtype=numpy.bool_)

    with pytest.raises(ValueError, match='does not match detector extent'):
        SimpleDiffractionDataset(metadata, contents_tree, [], wrong_shape)


def test_simple_diffraction_dataset_default_bad_pixels_matches_extent() -> None:
    metadata = DiffractionMetadata(
        num_patterns_per_array=[0],
        pattern_dtype=numpy.dtype(numpy.uint16),
        detector_extent=ImageExtent(width_px=16, height_px=8),
    )
    contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
    dataset = SimpleDiffractionDataset(metadata, contents_tree, [])
    assert dataset.get_bad_pixels().shape == (8, 16)
    assert not dataset.get_bad_pixels().any()


def test_create_unique_name_prevents_collision_after_insert() -> None:
    repo = _make_repository()
    repo.insert_dataset(repo.create_dataset('scan'))
    repo.insert_dataset(repo.create_dataset('scan'))
    assert [ds.get_name() for ds in repo] == ['scan', 'scan-1']


def _make_array(
    array_index: int,
    label: str,
    fill_value: float,
    num_patterns: int,
    detector_shape: tuple[int, int],
    index_offset: int,
) -> AssembledDiffractionArray:
    height, width = detector_shape
    patterns = numpy.full((num_patterns, height, width), fill_value, dtype=numpy.float64)
    indexes = numpy.arange(index_offset, index_offset + num_patterns, dtype=numpy.intp)
    data = AssembledDiffractionData(
        indexes=indexes,
        patterns=patterns,
        pixel_geometry=PixelGeometry(width_m=1.0, height_m=1.0),
        bad_pixels=numpy.zeros(detector_shape, dtype=numpy.bool_),
    )
    return AssembledDiffractionArray(array_index=array_index, label=label, data=data)


def test_dataset_average_pattern_is_weighted_mean_across_arrays() -> None:
    """Selecting a dataset node previews the average across all its patterns."""
    repo = _make_repository()
    dataset = repo.create_dataset('scan')
    repo.insert_dataset(dataset)

    # No arrays yet -> no preview to show.
    assert dataset.get_average_pattern() is None

    # Two arrays of different sizes with distinct uniform fills — the correct
    # weighted mean is (3*2 + 7*6) / (3 + 7) = 4.8 everywhere.
    dataset._insert_array(_make_array(0, 'a', 2.0, 3, (4, 4), index_offset=0))
    dataset._insert_array(_make_array(1, 'b', 6.0, 7, (4, 4), index_offset=3))

    result = dataset.get_average_pattern()
    assert result is not None
    numpy.testing.assert_allclose(result, numpy.full((4, 4), 4.8))
