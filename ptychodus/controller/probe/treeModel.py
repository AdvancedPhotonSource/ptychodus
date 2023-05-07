from __future__ import annotations
from typing import overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant


class ProbeTreeNode:

    def __init__(self, parentItem: ProbeTreeNode | None, name: str,
                 relativePowerPercent: int) -> None:
        # FIXME sizeInBytes
        self.parentItem = parentItem
        self._name = name
        self._relativePowerPercent = relativePowerPercent
        self.childItems: list[ProbeTreeNode] = list()

    @classmethod
    def createRoot(cls) -> ProbeTreeNode:
        return cls(None, str(), 0)

    def createChild(self, name: str, relativePowerPercent: int) -> ProbeTreeNode:
        childItem = ProbeTreeNode(self, name, relativePowerPercent)
        self.childItems.append(childItem)
        return childItem

    @property
    def name(self) -> str:
        return self._name

    @property
    def relativePowerPercent(self) -> int:
        return self._relativePowerPercent

    def row(self) -> int:
        if self.parentItem:
            return self.parentItem.childItems.index(self)

        return 0


class ProbeTreeModel(QAbstractItemModel):

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rootNode = ProbeTreeNode.createRoot()

    def setRootNode(self, rootNode: ProbeTreeNode) -> None:
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
                   role: int = Qt.DisplayRole) -> QVariant:
        value = None

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                value = QVariant('Mode')
            elif section == 1:
                value = QVariant('Relative Power')

        return QVariant(value)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = Qt.ItemFlags()

        if index.isValid():
            value = super().flags(index)

        return value

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parentItem = parent.internalPointer() if parent.isValid() else self._rootNode
            childItem = parentItem.childItems[row]

            if childItem:
                value = self.createIndex(row, column, childItem)

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            node = index.internalPointer()

            if role == Qt.DisplayRole and index.column() == 0:
                value = QVariant(node.name)
            elif role == Qt.UserRole and index.column() == 1:
                value = QVariant(node.relativePowerPercent)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.childItems)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2
