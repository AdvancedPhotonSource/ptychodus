from __future__ import annotations
from typing import Any, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush

from ptychodus.api.units import BYTES_PER_MEGABYTE

from ...model.product import ObjectAPI, ObjectRepository
from ...model.product.object import ObjectRepositoryItem


class ObjectTreeNode:
    def __init__(self, parent: ObjectTreeNode | None = None) -> None:
        self.parent = parent
        self.children: list[ObjectTreeNode] = list()

    def insert_node(self, index: int = -1) -> ObjectTreeNode:
        node = ObjectTreeNode(self)
        self.children.insert(index, node)
        return node

    def remove_node(self, index: int = -1) -> ObjectTreeNode:
        return self.children.pop(index)

    def row(self) -> int:
        return 0 if self.parent is None else self.parent.children.index(self)


class ObjectTreeModel(QAbstractItemModel):
    def __init__(
        self,
        repository: ObjectRepository,
        api: ObjectAPI,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._api = api
        self._editable_item_brush = editable_item_brush
        self._tree_root = ObjectTreeNode()
        self._header = [
            'Name',
            'Distance [m]',
            'Builder',
            'Data Type',
            'Width [px]',
            'Height [px]',
            'Size [MB]',
        ]

        for index, item in enumerate(repository):
            self.insert_item(index, item)

    @staticmethod
    def _append_layers(node: ObjectTreeNode, item: ObjectRepositoryItem) -> None:
        object_ = item.get_object()

        for layer in range(object_.num_layers):
            node.insert_node()

    def insert_item(self, index: int, item: ObjectRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        ObjectTreeModel._append_layers(self._tree_root.insert_node(index), item)
        self.endInsertRows()

    def update_item(self, index: int, item: ObjectRepositoryItem) -> None:
        top_left = self.index(index, 0)
        bottom_right = self.index(index, len(self._header))
        self.dataChanged.emit(top_left, bottom_right)

        node = self._tree_root.children[index]
        num_layers_old = len(node.children)
        num_layers_new = item.get_object().num_layers

        if num_layers_old < num_layers_new:
            self.beginInsertRows(top_left, num_layers_old, num_layers_new)

            while len(node.children) < num_layers_new:
                node.insert_node()

            self.endInsertRows()
        elif num_layers_old > num_layers_new:
            self.beginRemoveRows(top_left, num_layers_new, num_layers_old)

            while len(node.children) > num_layers_new:
                node.remove_node()

            self.endRemoveRows()

        child_top_left = self.index(0, 0, top_left)
        child_bottom_right = self.index(num_layers_new, len(self._header), top_left)
        self.dataChanged.emit(child_top_left, child_bottom_right)

    def remove_item(self, index: int, item: ObjectRepositoryItem) -> None:
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
    def parent(self, child: QModelIndex) -> QModelIndex: ...

    @overload
    def parent(self) -> QObject: ...

    def parent(self, child: QModelIndex | None = None) -> QModelIndex | QObject:
        if child is None:
            return super().parent()
        elif child.isValid():
            node = child.internalPointer()
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

            if role == Qt.ItemDataRole.DisplayRole:
                match index.column():
                    case 0:
                        return f'Layer {index.row() + 1}'
                    case 1:
                        try:
                            return item.layer_spacing_m[index.row()]
                        except IndexError:
                            return float('inf')
            elif role == Qt.ItemDataRole.BackgroundRole:
                if index.flags() & Qt.ItemFlag.ItemIsEditable:
                    return self._editable_item_brush
        else:
            item = self._repository[index.row()]
            object_ = item.get_object()

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                match index.column():
                    case 0:
                        return self._repository.get_name(index.row())
                    case 1:
                        return object_.get_total_thickness_m()
                    case 2:
                        return item.get_builder().get_name()
                    case 3:
                        return str(object_.dtype)
                    case 4:
                        return object_.width_px
                    case 5:
                        return object_.height_px
                    case 6:
                        return f'{object_.nbytes / BYTES_PER_MEGABYTE:.2f}'
            elif role == Qt.ItemDataRole.BackgroundRole:
                if index.flags() & Qt.ItemFlag.ItemIsEditable:
                    return self._editable_item_brush

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            parent = index.parent()

            if parent.isValid():
                if index.column() == 1:
                    item = self._repository[parent.row()]

                    if index.row() + 1 < item.get_num_layers():
                        value |= Qt.ItemFlag.ItemIsEditable
            else:
                if index.column() in (0, 2):
                    value |= Qt.ItemFlag.ItemIsEditable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            parent = index.parent()

            if parent.isValid():
                item = self._repository[parent.row()]

                if index.column() == 1:
                    try:
                        distance_m = float(value)
                    except ValueError:
                        return False

                    item.layer_spacing_m[index.row()] = distance_m
                    return False
            else:
                if index.column() == 0:
                    self._repository.set_name(index.row(), str(value))
                    return True
                elif index.column() == 2:
                    self._api.build_object(index.row(), str(value))
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.column() > 0:
            return 0

        node = parent.internalPointer() if parent.isValid() else self._tree_root
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)
