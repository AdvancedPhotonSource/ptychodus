"""Unit tests for ProductRepository.get_info_text."""

from __future__ import annotations

from unittest.mock import MagicMock

from ptychodus.model.product.repository import ProductRepository


def _make_item(nbytes: int, *, pending: bool = False) -> MagicMock:
    item = MagicMock()
    item.is_pending.return_value = pending
    item.get_product.return_value = MagicMock(nbytes=nbytes)
    return item


def test_empty_repository_reports_zero() -> None:
    assert ProductRepository().get_info_text() == 'Products: 0 [0 B]'


def test_info_text_counts_items_and_sums_bytes() -> None:
    repo = ProductRepository()
    repo.insert_product(_make_item(12_000_000))
    repo.insert_product(_make_item(340_000))

    assert repo.get_info_text() == 'Products: 2 [12.34 MB]'


def test_info_text_scales_the_unit() -> None:
    repo = ProductRepository()
    repo.insert_product(_make_item(4_100_000_000))

    assert repo.get_info_text() == 'Products: 1 [4.10 GB]'


def test_pending_items_are_counted_but_contribute_no_bytes() -> None:
    repo = ProductRepository()
    repo.insert_product(_make_item(1_000_000))
    repo.insert_product(_make_item(500_000_000, pending=True))

    assert repo.get_info_text() == 'Products: 2 [1.00 MB]'


def test_info_text_shrinks_after_removal() -> None:
    repo = ProductRepository()
    repo.insert_product(_make_item(1_000_000))
    repo.insert_product(_make_item(2_000_000))
    repo.remove_product(0)

    assert repo.get_info_text() == 'Products: 1 [2.00 MB]'
