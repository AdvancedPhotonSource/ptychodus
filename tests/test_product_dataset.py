"""Unit tests for per-product diffraction dataset association."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from ptychodus.model.processing.api import ProcessingAPI
from ptychodus.model.product.core import _DatasetOrphanObserver
from ptychodus.model.product.item import ProductRepositoryItem
from ptychodus.model.product.repository import ProductRepository
from ptychodus.model.product.item import ProductRepositoryObserver


class _RecordingObserver(ProductRepositoryObserver):
    def __init__(self) -> None:
        self.dataset_changed: list[int] = []

    def handle_item_inserted(self, index, item) -> None:  # noqa: ANN001
        pass

    def handle_metadata_changed(self, index, item) -> None:  # noqa: ANN001
        pass

    def handle_probe_positions_changed(self, index, item) -> None:  # noqa: ANN001
        pass

    def handle_probe_changed(self, index, item) -> None:  # noqa: ANN001
        pass

    def handle_object_changed(self, index, item) -> None:  # noqa: ANN001
        pass

    def handle_losses_changed(self, index, losses) -> None:  # noqa: ANN001
        pass

    def handle_dataset_changed(self, index, item) -> None:  # noqa: ANN001
        self.dataset_changed.append(index)

    def handle_state_changed(self, index, item) -> None:  # noqa: ANN001
        pass

    def handle_item_removed(self, index, item) -> None:  # noqa: ANN001
        pass


def test_repository_fans_out_dataset_changed() -> None:
    repo = ProductRepository()
    observer = _RecordingObserver()
    repo.add_observer(observer)

    item = MagicMock(spec=ProductRepositoryItem)
    item._index = 2
    item.get_name.return_value = 'product'

    repo.handle_dataset_changed(item)

    assert observer.dataset_changed == [2]


def test_repository_ignores_dataset_changed_for_unregistered_item() -> None:
    repo = ProductRepository()
    observer = _RecordingObserver()
    repo.add_observer(observer)

    item = MagicMock(spec=ProductRepositoryItem)
    item._index = -1
    item.get_name.return_value = 'orphan'

    repo.handle_dataset_changed(item)

    assert observer.dataset_changed == []


def test_orphan_observer_clears_only_matching_products() -> None:
    dataset = MagicMock()
    other_dataset = MagicMock()

    matching = MagicMock()
    matching.get_dataset.return_value = dataset
    unrelated = MagicMock()
    unrelated.get_dataset.return_value = other_dataset

    product_repository: Sequence[MagicMock] = [matching, unrelated]
    observer = _DatasetOrphanObserver(product_repository)  # type: ignore[arg-type]

    observer.handle_dataset_removed(0, dataset)

    matching.unbind_dataset.assert_called_once_with()
    unrelated.unbind_dataset.assert_not_called()


def _make_processing_api(item: MagicMock) -> ProcessingAPI:
    product_api = MagicMock()
    product_api.get_item.return_value = item
    return ProcessingAPI(
        MagicMock(),
        product_api,
        MagicMock(),
        MagicMock(),
        {},
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )


def test_get_reconstruct_input_raises_without_dataset() -> None:
    item = MagicMock()
    item.get_dataset.return_value = None
    item.get_name.return_value = 'product'
    api = _make_processing_api(item)

    with pytest.raises(RuntimeError):
        api.get_reconstruct_input(product_index=0)


def test_reconstruct_without_a_reconstructor_raises_before_inserting_a_product() -> None:
    """A library offering no algorithm must not leave a stray product behind."""
    product_api = MagicMock()
    algorithm = MagicMock()
    algorithm.get_current_reconstructor.return_value = None
    algorithm.get_current_parts.return_value = ('lib', '')
    api = ProcessingAPI(
        MagicMock(),
        product_api,
        MagicMock(),
        algorithm,
        {},
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    with pytest.raises(RuntimeError, match='lib offers no reconstructor'):
        api.reconstruct(product_index=0)

    product_api.insert_product.assert_not_called()


def test_get_reconstruct_input_uses_product_dataset() -> None:
    dataset = MagicMock()
    product = MagicMock()
    item = MagicMock()
    item.get_dataset.return_value = dataset
    item.get_product.return_value = product
    api = _make_processing_api(item)

    with patch('ptychodus.model.processing.api.prepare_reconstruct_input') as prepare:
        result = api.get_reconstruct_input(product_index=0)

    prepare.assert_called_once()
    assembled = dataset.get_assembled_data.return_value
    assert prepare.call_args.args[0] is assembled
    assert result is prepare.return_value
