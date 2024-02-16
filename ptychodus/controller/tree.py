from typing import overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant

from ..api.tree import SimpleTreeNode


class SimpleTreeModel(QAbstractItemModel):  # FIXME to patterns

    def __init__(self, rootNode: SimpleTreeNode, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rootNode = rootNode

    def setRootNode(self, rootNode: SimpleTreeNode) -> None:
        self.beginResetModel()
        self._rootNode = rootNode
        self.endResetModel()

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex:
        ...

    @overload
    def parent(self) -> QObject:
        ...

    def parent(self, child: QModelIndex | None = None) -> QModelIndex | QObject:
        if child is None:
            return super().parent()
        else:
            value = QModelIndex()

            if child.isValid():
                childItem = child.internalPointer()
                parentItem = childItem.parentItem

                if parentItem is self._rootNode:
                    value = QModelIndex()
                else:
                    value = self.createIndex(parentItem.row(), 0, parentItem)

            return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        value = None

        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            value = self._rootNode.data(section)

        return QVariant(value)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        return super().flags(index)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parentItem = parent.internalPointer() if parent.isValid() else self._rootNode
            childItem = parentItem.childItems[row]

            if childItem:
                value = self.createIndex(row, column, childItem)

        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            node = index.internalPointer()
            value = node.data(index.column())

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.childItems)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.itemData)
