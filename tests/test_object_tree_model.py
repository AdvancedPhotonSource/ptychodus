"""Regression tests for ObjectTreeModel's Qt model bookkeeping.

Four defects are covered here. Qt's beginInsertRows/beginRemoveRows take an
*inclusive* last row, so announcing `num_layers_new` rather than
`num_layers_new - 1` claims one more row than is actually inserted or removed.
index() rejects a column equal to columnCount(), so building a dataChanged
range against `len(self._header)` yields an invalid index and a malformed
signal. setData returned False after successfully writing a layer distance,
telling Qt the edit had failed. And the last layer, which has no following
layer to be spaced from, displayed `inf` instead of a blank cell.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip('PyQt5')

from PyQt5.QtCore import QModelIndex, Qt  # noqa: E402
from PyQt5.QtGui import QBrush  # noqa: E402

from ptychodus.controller.object.tree_model import ObjectTreeModel, ObjectTreeNode  # noqa: E402


def _make_item(num_layers: int) -> MagicMock:
    """An ObjectRepositoryItem whose object reports *num_layers* layers."""
    spacing = [1e-6] * (num_layers - 1)

    object_ = MagicMock()
    object_.num_layers = num_layers

    item = MagicMock()
    item.get_object.return_value = object_
    item.get_num_layers.return_value = num_layers
    item.layer_spacing_m = spacing
    return item


def _make_model(items: list[MagicMock]) -> ObjectTreeModel:
    repository = MagicMock()
    repository.__len__.return_value = len(items)
    repository.__getitem__.side_effect = lambda i: items[i]
    repository.__iter__.side_effect = lambda: iter(items)
    return ObjectTreeModel(repository, MagicMock(), QBrush())


class TestObjectTreeNode:
    def test_insert_node_appends_in_order(self) -> None:
        """insert_node() used to default to list.insert(-1, ...), which inserts
        before the last child instead of appending."""
        root = ObjectTreeNode()
        children = [root.insert_node() for _ in range(4)]

        assert root.children == children
        assert [node.row() for node in children] == [0, 1, 2, 3]

    def test_insert_node_at_explicit_index(self) -> None:
        root = ObjectTreeNode()
        first = root.insert_node()
        inserted = root.insert_node(0)

        assert root.children == [inserted, first]


class TestUpdateItemRowCounts:
    def test_growing_layers_inserts_exact_row_count(self) -> None:
        items = [_make_item(1)]
        model = _make_model(items)
        parent = model.index(0, 0)
        assert model.rowCount(parent) == 1

        items[0] = _make_item(3)
        model.update_item(0, items[0])

        assert model.rowCount(model.index(0, 0)) == 3

    def test_shrinking_layers_removes_exact_row_count(self) -> None:
        items = [_make_item(4)]
        model = _make_model(items)
        assert model.rowCount(model.index(0, 0)) == 4

        items[0] = _make_item(2)
        model.update_item(0, items[0])

        assert model.rowCount(model.index(0, 0)) == 2

    @pytest.mark.parametrize(('old', 'new'), [(1, 3), (3, 1), (2, 2), (1, 9), (9, 1)])
    def test_row_signals_announce_the_rows_actually_changed(self, old: int, new: int) -> None:
        """The inclusive-bound bug made Qt's own model consistency check fail:
        rowsInserted reported a range wider than rowCount ended up being."""
        items = [_make_item(old)]
        model = _make_model(items)

        inserted: list[tuple[int, int]] = []
        removed: list[tuple[int, int]] = []
        model.rowsInserted.connect(lambda _p, f, l: inserted.append((f, l)))
        model.rowsRemoved.connect(lambda _p, f, l: removed.append((f, l)))

        items[0] = _make_item(new)
        model.update_item(0, items[0])

        assert model.rowCount(model.index(0, 0)) == new

        if new > old:
            assert inserted == [(old, new - 1)]
            assert removed == []
        elif new < old:
            assert removed == [(new, old - 1)]
            assert inserted == []
        else:
            assert inserted == [] and removed == []

    def test_data_changed_ranges_are_valid(self) -> None:
        """index() rejects a column equal to columnCount(), so the old
        `len(self._header)` bound produced an invalid bottom-right index."""
        items = [_make_item(3)]
        model = _make_model(items)

        ranges: list[tuple[QModelIndex, QModelIndex]] = []
        model.dataChanged.connect(lambda tl, br, roles=None: ranges.append((tl, br)))

        model.update_item(0, items[0])

        assert ranges
        last_column = model.columnCount() - 1

        for top_left, bottom_right in ranges:
            assert top_left.isValid()
            assert bottom_right.isValid()
            assert bottom_right.column() == last_column
            assert top_left.parent() == bottom_right.parent()
            assert top_left.row() <= bottom_right.row()


class TestLayerDistanceCell:
    def test_last_layer_distance_is_blank(self) -> None:
        """N layers have N-1 spacings; the final layer has no distance."""
        model = _make_model([_make_item(3)])
        parent = model.index(0, 0)

        assert model.data(model.index(0, 1, parent)) == pytest.approx(1e-6)
        assert model.data(model.index(1, 1, parent)) == pytest.approx(1e-6)
        assert model.data(model.index(2, 1, parent)) is None

    def test_last_layer_distance_is_not_editable(self) -> None:
        model = _make_model([_make_item(3)])
        parent = model.index(0, 0)

        assert model.flags(model.index(0, 1, parent)) & Qt.ItemFlag.ItemIsEditable
        assert not (model.flags(model.index(2, 1, parent)) & Qt.ItemFlag.ItemIsEditable)


class TestSetData:
    def test_successful_layer_distance_edit_returns_true(self) -> None:
        """setData wrote the value but returned False, so Qt treated the edit as
        rejected and skipped its own dataChanged emission."""
        items = [_make_item(3)]
        model = _make_model(items)
        parent = model.index(0, 0)

        assert model.setData(model.index(0, 1, parent), '2.5e-6') is True
        assert items[0].layer_spacing_m[0] == pytest.approx(2.5e-6)

    def test_unparseable_layer_distance_returns_false(self) -> None:
        items = [_make_item(3)]
        model = _make_model(items)
        parent = model.index(0, 0)

        assert model.setData(model.index(0, 1, parent), 'not a number') is False
        assert items[0].layer_spacing_m[0] == pytest.approx(1e-6)
