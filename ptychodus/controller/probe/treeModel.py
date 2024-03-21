from __future__ import annotations
from typing import Any, overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ...model.product import ProbeRepository
from ...model.product.probe import ProbeRepositoryItem


class ProbeTreeNode:

    def __init__(self, parent: ProbeTreeNode | None = None) -> None:
        self.parent = parent
        self.children: list[ProbeTreeNode] = list()

    def insertNode(self, index: int = -1) -> ProbeTreeNode:
        node = ProbeTreeNode(self)
        self.children.insert(index, node)
        return node

    def removeNode(self, index: int = -1) -> ProbeTreeNode:
        return self.children.pop(index)

    def row(self) -> int:
        return 0 if self.parent is None else self.parent.children.index(self)


class ProbeTreeModel(QAbstractItemModel):

    def __init__(self, repository: ProbeRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository
        self._treeRoot = ProbeTreeNode()
        self._header = [
            'Name', 'Relative Power', 'Builder', 'Data Type', 'Width [px]', 'Height [px]',
            'Size [MB]'
        ]

    @staticmethod
    def _appendModes(node: ProbeTreeNode, item: ProbeRepositoryItem) -> None:
        object_ = item.getProbe()

        for layer in range(object_.numberOfModes):
            node.insertNode()

    def insertItem(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        ProbeTreeModel._appendModes(self._treeRoot.insertNode(index), item)
        self.endInsertRows()

    def updateItem(self, index: int, item: ProbeRepositoryItem) -> None:
        topLeft = self.index(index, 0)
        bottomRight = self.index(index, len(self._header))
        self.dataChanged.emit(topLeft, bottomRight)

        node = self._treeRoot.children[index]
        numModesOld = len(node.children)
        numModesNew = item.getProbe().numberOfModes

        if numModesOld < numModesNew:
            self.beginInsertRows(topLeft, numModesOld, numModesNew)

            while len(node.children) < numModesNew:
                node.insertNode()

            self.endInsertRows()
        elif numModesOld > numModesNew:
            self.beginRemoveRows(topLeft, numModesNew, numModesOld)

            while len(node.children) > numModesNew:
                node.removeNode()

            self.endRemoveRows()

        childTopLeft = self.index(0, 0, topLeft)
        childBottomRight = self.index(numModesNew, len(self._header), topLeft)
        self.dataChanged.emit(childTopLeft, childBottomRight)

    def removeItem(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self._treeRoot.removeNode(index)
        self.endRemoveRows()

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    @overload
    def parent(self, index: QModelIndex) -> QModelIndex:
        ...

    @overload
    def parent(self) -> QObject:
        ...

    def parent(self, index: QModelIndex | None = None) -> QModelIndex | QObject:
        if index is None:
            return super().parent()
        elif index.isValid():
            node = index.internalPointer()
            parentNode = node.parent
            return QModelIndex() if parentNode is self._treeRoot \
                    else self.createIndex(parentNode.row(), 0, parentNode)

        return QModelIndex()

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if self.hasIndex(row, column, parent):
            if parent.isValid():
                parentNode = parent.internalPointer()
                node = parentNode.children[row]
            else:
                node = self._treeRoot.children[row]

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
                probe = item.getProbe()

                try:
                    relativePower = probe.getModeRelativePower(index.row())
                except IndexError:
                    return -1

                if numpy.isfinite(relativePower):
                    return int(100. * relativePower)
        else:
            item = self._repository[index.row()]
            probe = item.getProbe()

            if role == Qt.ItemDataRole.DisplayRole:
                if index.column() == 0:
                    return self._repository.getName(index.row())
                elif index.column() == 1:
                    return None
                elif index.column() == 2:
                    return item.getBuilder().getName()
                elif index.column() == 3:
                    return str(probe.dataType)
                elif index.column() == 4:
                    return probe.widthInPixels
                elif index.column() == 5:
                    return probe.heightInPixels
                elif index.column() == 6:
                    return f'{probe.sizeInBytes / (1024 * 1024):.2f}'
            elif role == Qt.ItemDataRole.UserRole and index.column() == 1:
                probe = item.getProbe()
                coherence = probe.getCoherence()
                return int(100. * coherence) if numpy.isfinite(coherence) else -1

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            parent = index.parent()

            if not parent.isValid() and index.column() in (0, 2):
                value |= Qt.ItemFlag.ItemIsEditable

        return value

    def setData(self,
                index: QModelIndex,
                value: Any,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            parent = index.parent()

            if not parent.isValid():
                if index.column() == 0:
                    self._repository.setName(index.row(), str(value))
                    return True
                elif index.column() == 2:
                    self._repository.setBuilderByName(index.row(), str(value))
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = parent.internalPointer() if parent.isValid() else self._treeRoot
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
