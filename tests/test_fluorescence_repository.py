"""Unit tests for FluorescenceRepository sizing and info text."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy

from ptychodus.api.fluorescence import ElementMap, FluorescenceDataset
from ptychodus.model.fluorescence.repository import (
    FluorescenceRepository,
    FluorescenceRepositoryItem,
)


def _make_dataset(num_maps: int, shape: tuple[int, int]) -> FluorescenceDataset:
    element_maps = [
        ElementMap(f'E{index}', numpy.zeros(shape, dtype=numpy.float32))
        for index in range(num_maps)
    ]
    return FluorescenceDataset(
        _element_maps=element_maps,
        counts_per_second_path='/counts',
        channel_names_path='/names',
    )


def _make_item(measured: FluorescenceDataset, *, name: str = 'item') -> FluorescenceRepositoryItem:
    return FluorescenceRepositoryItem(
        MagicMock(),
        name=name,
        product=MagicMock(),
        measured=measured,
    )


def _make_named_item(name: str) -> FluorescenceRepositoryItem:
    return _make_item(_make_dataset(1, (2, 2)), name=name)


def test_item_nbytes_covers_the_measured_dataset() -> None:
    measured = _make_dataset(2, (10, 10))
    assert _make_item(measured).get_nbytes() == measured.nbytes


def test_item_nbytes_grows_with_the_enhanced_dataset() -> None:
    measured = _make_dataset(2, (10, 10))
    enhanced = _make_dataset(2, (20, 20))
    item = _make_item(measured)

    before = item.get_nbytes()
    item.set_enhanced(enhanced)

    assert before == measured.nbytes
    assert item.get_nbytes() == measured.nbytes + enhanced.nbytes


def test_summary_caches_do_not_contribute() -> None:
    measured = _make_dataset(2, (10, 10))
    item = _make_item(measured)

    # Populating the cache must not change the reported size: leaves have to sum
    # exactly to their item, and items to the repository total.
    assert item.get_measured_summary() is not None
    assert item.get_nbytes() == measured.nbytes


def test_empty_repository_reports_zero() -> None:
    assert FluorescenceRepository(MagicMock()).get_info_text() == 'Datasets: 0 [0 B]'


def test_info_text_counts_items_and_sums_bytes() -> None:
    repo = FluorescenceRepository(MagicMock())
    # 2 maps of 500 float32 each == 4000 B per item.
    repo.insert_item(_make_item(_make_dataset(2, (25, 20))))
    repo.insert_item(_make_item(_make_dataset(2, (25, 20))))

    assert repo.get_info_text() == 'Datasets: 2 [8.00 kB]'


def test_info_text_shrinks_after_removal() -> None:
    repo = FluorescenceRepository(MagicMock())
    repo.insert_item(_make_item(_make_dataset(2, (25, 20))))
    repo.insert_item(_make_item(_make_dataset(2, (25, 20))))
    repo.remove_item(0)

    assert repo.get_info_text() == 'Datasets: 1 [4.00 kB]'


def test_create_unique_name_returns_input_when_free() -> None:
    assert FluorescenceRepository(MagicMock()).create_unique_name('foo') == 'foo'


def test_create_unique_name_suffixes_collisions() -> None:
    repo = FluorescenceRepository(MagicMock())
    repo.insert_item(_make_named_item('foo'))
    repo.insert_item(_make_named_item('foo-1'))

    assert repo.create_unique_name('foo') == 'foo-2'


def test_create_unique_name_increments_rather_than_nests() -> None:
    repo = FluorescenceRepository(MagicMock())
    repo.insert_item(_make_named_item('foo'))
    repo.insert_item(_make_named_item('foo-1'))

    assert repo.create_unique_name('foo-1') == 'foo-2'


def test_create_unique_name_maps_empty_to_unnamed() -> None:
    assert FluorescenceRepository(MagicMock()).create_unique_name('') == 'Unnamed'
