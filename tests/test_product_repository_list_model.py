"""Tests for ProductRepositoryComboProxyModel.flags — non-READY items must be
disabled so pending / failed stubs cannot be selected in the Processing tab's
product picker (or any other combo bound to this proxy).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip('PyQt5')

from PyQt5.QtCore import QModelIndex, Qt  # noqa: E402
from PyQt5.QtGui import QBrush  # noqa: E402

from ptychodus.controller.product.core import (  # noqa: E402
    ProductRepositoryComboProxyModel,
    ProductRepositoryTableModel,
)


def _make_item(*, pending: bool = False, failed: bool = False) -> MagicMock:
    item = MagicMock()
    item.is_pending.return_value = pending
    item.is_failed.return_value = failed
    item.get_name.return_value = 'p'
    metadata = MagicMock()
    metadata.name.get_value.return_value = 'p'
    item.get_metadata_item.return_value = metadata
    return item


def _make_proxy(items: list[MagicMock]) -> ProductRepositoryComboProxyModel:
    repo = MagicMock()
    repo.__len__.return_value = len(items)
    repo.__getitem__.side_effect = lambda i: items[i]
    source = ProductRepositoryTableModel(repo, QBrush())
    return ProductRepositoryComboProxyModel(source, repo)


def test_flags_ready_item_uses_default_flags() -> None:
    proxy = _make_proxy([_make_item()])
    index = proxy.index(0, 0, QModelIndex())

    flags = proxy.flags(index)

    assert flags & Qt.ItemIsSelectable
    assert flags & Qt.ItemIsEnabled
    # Editability must be stripped so comboboxes don't offer inline edits on
    # the shared source model.
    assert not (flags & Qt.ItemIsEditable)


def test_flags_pending_item_returns_no_flags() -> None:
    proxy = _make_proxy([_make_item(pending=True)])
    index = proxy.index(0, 0, QModelIndex())

    flags = proxy.flags(index)

    assert not (flags & Qt.ItemIsSelectable)
    assert not (flags & Qt.ItemIsEnabled)


def test_flags_failed_item_returns_no_flags() -> None:
    proxy = _make_proxy([_make_item(failed=True)])
    index = proxy.index(0, 0, QModelIndex())

    flags = proxy.flags(index)

    assert not (flags & Qt.ItemIsSelectable)
    assert not (flags & Qt.ItemIsEnabled)


def test_flags_invalid_index_returns_default() -> None:
    proxy = _make_proxy([_make_item(pending=True)])

    # Should not crash and should not consult the repository.
    flags = proxy.flags(QModelIndex())

    assert isinstance(flags, Qt.ItemFlags)


def test_column_count_is_one() -> None:
    proxy = _make_proxy([_make_item(), _make_item()])
    assert proxy.columnCount() == 1
