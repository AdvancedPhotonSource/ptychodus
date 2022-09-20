from typing import overload, Optional, Union

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant

from ..api.tree import SimpleTreeNode


class SimpleTreeModel(QAbstractItemModel):

    def __init__(self, rootNode: SimpleTreeNode, parent: QObject = None) -> None:
        super().__init__(parent)
        self._rootNode = rootNode

    def setRootNode(self, rootNode: SimpleTreeNode) -> None:
        self.beginResetModel()
        self._rootNode = rootNode
        self.endResetModel()

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid() and role == Qt.DisplayRole:
            node = index.internalPointer()
            value = node.data(index.column())

        return value

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = Qt.ItemFlags()

        if index.isValid():
            value = super().flags(index)

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        value = None

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            value = self._rootNode.data(section)

        return QVariant(value)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parentItem = parent.internalPointer() if parent.isValid() else self._rootNode
            childItem = parentItem.childItems[row]

            if childItem:
                value = self.createIndex(row, column, childItem)

        return value

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex:
        ...

    @overload
    def parent(self) -> QObject:
        ...

    def parent(self, child: Optional[QModelIndex] = None) -> Union[QModelIndex, QObject]:
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

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        parentItem = parent.internalPointer() if parent.isValid() else self._rootNode

        return len(parentItem.childItems)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(parent.internalPointer().itemData) if parent.isValid() \
                else len(self._rootNode.itemData)
