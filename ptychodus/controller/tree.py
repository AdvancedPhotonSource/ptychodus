from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..model import SimpleTreeNode


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
            item = index.internalPointer()
            value = item.data(index.column())

        return value

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = Qt.NoItemFlags

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

        return value

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parentItem = parent.internalPointer() if parent.isValid() else self._rootNode
            childItem = parentItem.childItems[row]

            if childItem:
                value = self.createIndex(row, column, childItem)

        return value

    def parent(self, index: QModelIndex) -> QModelIndex:
        value = QModelIndex()

        if index.isValid():
            childItem = index.internalPointer()
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
