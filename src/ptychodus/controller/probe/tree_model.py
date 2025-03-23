from __future__ import annotations
from typing import Any, overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ...model.product import ProbeAPI, ProbeRepository
from ...model.product.probe import ProbeRepositoryItem


class ProbeTreeNode:
    def __init__(self, parent: ProbeTreeNode | None = None) -> None:
        self.parent = parent
        self.children: list[ProbeTreeNode] = list()

    def insert_node(self, index: int = -1) -> ProbeTreeNode:
        node = ProbeTreeNode(self)
        self.children.insert(index, node)
        return node

    def remove_node(self, index: int = -1) -> ProbeTreeNode:
        return self.children.pop(index)

    def row(self) -> int:
        return 0 if self.parent is None else self.parent.children.index(self)


class ProbeTreeModel(QAbstractItemModel):
    def __init__(
        self, repository: ProbeRepository, api: ProbeAPI, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._api = api
        self._tree_root = ProbeTreeNode()
        self._header = [
            'Name',
            'Relative Power',
            'Builder',
            'Data Type',
            'Width [px]',
            'Height [px]',
            'Size [MB]',
        ]

        for index, item in enumerate(repository):
            self.insert_item(index, item)

    @staticmethod
    def _append_modes(node: ProbeTreeNode, item: ProbeRepositoryItem) -> None:
        probe = item.get_probe()

        for layer in range(probe.num_incoherent_modes):
            node.insert_node()

    def insert_item(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        ProbeTreeModel._append_modes(self._tree_root.insert_node(index), item)
        self.endInsertRows()

    def update_item(self, index: int, item: ProbeRepositoryItem) -> None:
        top_left = self.index(index, 0)
        bottom_right = self.index(index, len(self._header))
        self.dataChanged.emit(top_left, bottom_right)

        node = self._tree_root.children[index]
        num_modes_old = len(node.children)
        num_modes_new = item.get_probe().num_incoherent_modes

        if num_modes_old < num_modes_new:
            self.beginInsertRows(top_left, num_modes_old, num_modes_new)

            while len(node.children) < num_modes_new:
                node.insert_node()

            self.endInsertRows()
        elif num_modes_old > num_modes_new:
            self.beginRemoveRows(top_left, num_modes_new, num_modes_old)

            while len(node.children) > num_modes_new:
                node.remove_node()

            self.endRemoveRows()

        child_top_left = self.index(0, 0, top_left)
        child_bottom_right = self.index(num_modes_new, len(self._header), top_left)
        self.dataChanged.emit(child_top_left, child_bottom_right)

    def remove_item(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self._tree_root.remove_node(index)
        self.endRemoveRows()

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    @overload
    def parent(self, index: QModelIndex) -> QModelIndex: ...

    @overload
    def parent(self) -> QObject: ...

    def parent(self, index: QModelIndex | None = None) -> QModelIndex | QObject:
        if index is None:
            return super().parent()
        elif index.isValid():
            node = index.internalPointer()
            parent_node = node.parent
            return (
                QModelIndex()
                if parent_node is self._tree_root
                else self.createIndex(parent_node.row(), 0, parent_node)
            )

        return QModelIndex()

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if self.hasIndex(row, column, parent):
            if parent.isValid():
                parent_node = parent.internalPointer()
                node = parent_node.children[row]
            else:
                node = self._tree_root.children[row]

            return self.createIndex(row, column, node)

        return QModelIndex()

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        parent = index.parent()

        if parent.isValid():
            item = self._repository[parent.row()]

            if role == Qt.ItemDataRole.DisplayRole and index.column() == 0:
                return f'Mode {index.row() + 1}'
            elif role == Qt.ItemDataRole.UserRole and index.column() == 1:
                probe = item.get_probe()

                try:
                    relative_power = probe.get_incoherent_mode_relative_power(index.row())
                except IndexError:
                    return -1

                if numpy.isfinite(relative_power):
                    return int(100.0 * relative_power)
        else:
            item = self._repository[index.row()]
            probe = item.get_probe()

            if role == Qt.ItemDataRole.DisplayRole:
                if index.column() == 0:
                    return self._repository.get_name(index.row())
                elif index.column() == 1:
                    return None
                elif index.column() == 2:
                    return item.get_builder().get_name()
                elif index.column() == 3:
                    return str(probe.dtype)
                elif index.column() == 4:
                    return probe.width_px
                elif index.column() == 5:
                    return probe.height_px
                elif index.column() == 6:
                    return f'{probe.nbytes / (1024 * 1024):.2f}'
            elif role == Qt.ItemDataRole.UserRole and index.column() == 1:
                probe = item.get_probe()
                coherence = probe.get_coherence()
                return int(100.0 * coherence) if numpy.isfinite(coherence) else -1

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            parent = index.parent()

            if not parent.isValid() and index.column() in (0, 2):
                value |= Qt.ItemFlag.ItemIsEditable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            parent = index.parent()

            if not parent.isValid():
                if index.column() == 0:
                    self._repository.set_name(index.row(), str(value))
                    return True
                elif index.column() == 2:
                    self._api.buildProbe(index.row(), str(value))
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.column() > 0:
            return 0

        node = parent.internalPointer() if parent.isValid() else self._tree_root
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)
