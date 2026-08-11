from __future__ import annotations
from enum import Enum
from typing import Any, cast, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush, QFont

from ptychodus.api.common import RealArrayType
from ptychodus.api.fluorescence import ElementMap, FluorescenceDataset

from ...model.fluorescence import (
    FluorescenceItemState,
    FluorescenceRepository,
    FluorescenceRepositoryItem,
    FluorescenceRepositoryObserver,
)


class DisplayMode(Enum):
    MEASURED = 'measured'
    ENHANCED = 'enhanced'


_COL_NAME = 0
_COL_COUNTS = 3


def _select_display_quantity(item: FluorescenceRepositoryItem, display_mode: DisplayMode) -> FluorescenceDataset:
    """Return the requested display quantity, falling back to measured when enhanced is absent.

    The tree walks measured element names as the canonical identity, so this
    fallback keeps leaf rows renderable even when the enhancement hasn't been
    run yet or when the enhanced dataset happens to be missing an element by
    name.
    """
    if display_mode is DisplayMode.ENHANCED:
        enhanced = item.get_enhanced()
        if enhanced is not None:
            return enhanced
    return item.get_measured()


def _lookup_display_element(
    item: FluorescenceRepositoryItem, element_name: str, display: DisplayMode
) -> ElementMap | None:
    dataset = _select_display_quantity(item, display)
    for element_map in dataset.element_maps:
        if element_map.name == element_name:
            return element_map
    return None


class _TreeNode:
    """Base tree node — item root or element leaf."""

    def __init__(self, parent: _TreeNode | None) -> None:
        self.parent = parent
        self.children: list[_TreeNode] = []

    def row(self) -> int:
        return 0 if self.parent is None else self.parent.children.index(self)

    def get_data(
        self, item: FluorescenceRepositoryItem, display: DisplayMode
    ) -> RealArrayType | None:
        return None

    def get_counts(self, item: FluorescenceRepositoryItem, display: DisplayMode) -> float | None:
        return None


class _ItemNode(_TreeNode):
    def __init__(self, parent: _TreeNode) -> None:
        super().__init__(parent)

    def get_data(
        self, item: FluorescenceRepositoryItem, display: DisplayMode
    ) -> RealArrayType | None:
        if display is DisplayMode.ENHANCED:
            enhanced = item.get_enhanced_summary()
            if enhanced is not None:
                return enhanced
            return item.get_measured_summary()
        return item.get_measured_summary()

    def get_counts(self, item: FluorescenceRepositoryItem, display: DisplayMode) -> float | None:
        summary = self.get_data(item, display)
        if summary is None:
            return None
        return float(summary.sum())


class _ElementNode(_TreeNode):
    def __init__(self, parent: _ItemNode, element_index: int) -> None:
        super().__init__(parent)
        self.element_index = element_index

    def _measured_element(self, item: FluorescenceRepositoryItem) -> ElementMap | None:
        maps = item.get_measured().element_maps
        if 0 <= self.element_index < len(maps):
            return maps[self.element_index]
        return None

    def element_name(self, item: FluorescenceRepositoryItem) -> str:
        measured = self._measured_element(item)
        return measured.name if measured is not None else ''

    def _resolve(self, item: FluorescenceRepositoryItem, display: DisplayMode) -> ElementMap | None:
        measured = self._measured_element(item)
        if measured is None:
            return None
        if display is DisplayMode.MEASURED:
            return measured
        # Enhanced by name — fall back to measured when the enhanced dataset
        # is absent or lacks a matching name.
        return _lookup_display_element(item, measured.name, DisplayMode.ENHANCED) or measured

    def get_data(
        self, item: FluorescenceRepositoryItem, display: DisplayMode
    ) -> RealArrayType | None:
        element = self._resolve(item, display)
        return None if element is None else element.counts_per_second

    def get_counts(self, item: FluorescenceRepositoryItem, display: DisplayMode) -> float | None:
        element = self._resolve(item, display)
        if element is None:
            return None
        return float(element.counts_per_second.sum())


class FluorescenceRepositoryTreeModel(QAbstractItemModel):
    """Two-level tree over FluorescenceRepository.

    - Level 1: `_ItemNode` per repository item (top-level rows).
    - Level 2: `_ElementNode` per element in the item's measured dataset
      (leaves).

    A single `display` field (measured/enhanced) drives the Counts column and
    the array returned by ``get_data``. The controller flips it via
    ``set_display`` when the user toggles the panel-level radio.

    Duck-typed against FluorescenceRepositoryObserver — inheriting the ABC
    would clash with sip's wrappertype metaclass on QAbstractItemModel.
    """

    _HEADER = ('Name', 'Product', 'Elements', 'Counts')

    def __init__(
        self,
        repository: FluorescenceRepository,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._root = _TreeNode(None)
        self._display = DisplayMode.MEASURED

        for item in repository:
            self._root.children.append(self._build_item_node(item))

        repository.add_observer(cast(FluorescenceRepositoryObserver, self))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_display(self, display: DisplayMode) -> None:
        if display is self._display:
            return
        self._display = display
        self._broadcast_counts_and_state()

    def get_display(self) -> DisplayMode:
        return self._display

    def item_row_for_index(self, index: QModelIndex) -> int:
        """Repository row index for whichever node ``index`` belongs to; -1 if none."""
        if not index.isValid():
            return -1
        node = index.internalPointer()
        if isinstance(node, _ItemNode):
            return node.row()
        if isinstance(node, _ElementNode):
            parent = node.parent
            return parent.row() if parent is not None else -1
        return -1

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _build_item_node(self, item: FluorescenceRepositoryItem) -> _ItemNode:
        item_node = _ItemNode(self._root)
        for element_index in range(len(item.get_measured().element_maps)):
            item_node.children.append(_ElementNode(item_node, element_index))
        return item_node

    def _broadcast_counts_and_state(self) -> None:
        """Refresh Counts across the whole tree after a display switch."""
        num_rows = self.rowCount()
        if num_rows == 0:
            return
        # Top-level rows: Counts follows the display.
        top_left = self.index(0, _COL_COUNTS)
        bottom_right = self.index(num_rows - 1, _COL_COUNTS)
        self.dataChanged.emit(top_left, bottom_right)
        # Element leaves: Counts likewise.
        for item_row in range(num_rows):
            parent_index = self.index(item_row, 0)
            num_children = self.rowCount(parent_index)
            if num_children == 0:
                continue
            leaf_top = self.index(0, _COL_COUNTS, parent_index)
            leaf_bottom = self.index(num_children - 1, _COL_COUNTS, parent_index)
            self.dataChanged.emit(leaf_top, leaf_bottom)

    # ------------------------------------------------------------------
    # Repository observer callbacks
    # ------------------------------------------------------------------

    def handle_item_inserted(self, index: int, item: FluorescenceRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self._root.children.insert(index, self._build_item_node(item))
        self.endInsertRows()

    def handle_item_removed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        del self._root.children[index]
        self.endRemoveRows()

    def handle_metadata_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        self._emit_row_changed(index)

    def handle_enhanced_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        # Structural change: none (the enhanced dataset is a display, not a
        # separate branch). Refresh Counts for the item row and its leaves,
        # so the current display's numbers stay in sync.
        self._emit_row_changed(index)
        parent_idx = self.index(index, 0)
        num_children = self.rowCount(parent_idx)
        if num_children > 0:
            top_left = self.index(0, _COL_COUNTS, parent_idx)
            bottom_right = self.index(num_children - 1, _COL_COUNTS, parent_idx)
            self.dataChanged.emit(top_left, bottom_right)

    def handle_state_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        self._emit_row_changed(
            index,
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.FontRole,
                Qt.ItemDataRole.ForegroundRole,
            ],
        )

    def _emit_row_changed(self, index: int, roles: list[int] | None = None) -> None:
        top_left = self.index(index, 0)
        bottom_right = self.index(index, self.columnCount() - 1)
        if roles is None:
            self.dataChanged.emit(top_left, bottom_right)
        else:
            self.dataChanged.emit(top_left, bottom_right, roles)

    # ------------------------------------------------------------------
    # QAbstractItemModel plumbing
    # ------------------------------------------------------------------

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._HEADER[section]

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...

    @overload
    def parent(self) -> QObject: ...

    def parent(self, child: QModelIndex | None = None) -> QModelIndex | QObject:
        if child is None:
            return super().parent()
        if not child.isValid():
            return QModelIndex()

        node = child.internalPointer()
        if node is None or node.parent is None or node.parent is self._root:
            return QModelIndex()
        return self.createIndex(node.parent.row(), 0, node.parent)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node = parent.internalPointer() if parent.isValid() else self._root
        try:
            node = parent_node.children[row]
        except IndexError:
            return QModelIndex()
        return self.createIndex(row, column, node)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.column() > 0:
            return 0
        parent_node = parent.internalPointer() if parent.isValid() else self._root
        return len(parent_node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._HEADER)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)
        if not index.isValid():
            return value

        node = index.internalPointer()
        if not isinstance(node, _ItemNode):
            return value
        if index.column() != _COL_NAME:
            return value

        try:
            item = self._repository[index.row()]
        except IndexError:
            return value
        # ORPHANED items can still be renamed — label is metadata, not tied to
        # product lifetime; ENHANCING items lock the label so the user doesn't
        # rename a row mid-task.
        if item.get_state() is not FluorescenceItemState.ENHANCING:
            value |= Qt.ItemFlag.ItemIsEditable
        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        node = index.internalPointer()
        if node is None:
            return None

        item_row = self.item_row_for_index(index)
        if item_row < 0:
            return None
        try:
            item = self._repository[item_row]
        except IndexError:
            return None

        state = item.get_state()

        if isinstance(node, _ElementNode):
            if role == Qt.ItemDataRole.DisplayRole:
                match index.column():
                    case 0:
                        return node.element_name(item)
                    case 3:
                        counts = node.get_counts(item, self._display)
                        return _format_counts(counts)
                    case _:
                        return None
            elif role == Qt.ItemDataRole.FontRole:
                font = QFont()
                font.setItalic(True)
                return font
            return None

        # Item row.
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            match index.column():
                case 0:
                    return item.get_label()
                case 1:
                    return item.get_product().get_name()
                case 2:
                    return len(item.get_measured().element_maps)
                case 3:
                    counts = node.get_counts(item, self._display)
                    return _format_counts(counts)
        elif role == Qt.ItemDataRole.FontRole:
            if state is FluorescenceItemState.ENHANCING:
                font = QFont()
                font.setItalic(True)
                return font
            if state is FluorescenceItemState.FAILED or state is FluorescenceItemState.ORPHANED:
                font = QFont()
                font.setStrikeOut(True)
                return font
        elif role == Qt.ItemDataRole.ForegroundRole:
            if state is not FluorescenceItemState.READY:
                return QBrush(Qt.GlobalColor.gray)
        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        node = index.internalPointer()
        if not isinstance(node, _ItemNode) or index.column() != _COL_NAME:
            return False
        try:
            item = self._repository[index.row()]
        except IndexError:
            return False
        item.set_label(str(value))
        return True


def _format_counts(counts: float | None) -> str:
    if counts is None:
        return '—'
    return f'{counts:.4g}'
