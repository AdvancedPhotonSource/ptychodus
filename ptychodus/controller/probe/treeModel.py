from __future__ import annotations
from typing import overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant

from ...api.probe import ProbeArrayType
from ...model.probe import ProbeRepositoryItemPresenter


class ProbeTreeNode:

    def __init__(self, parentItem: ProbeTreeNode | None,
                 presenter: ProbeRepositoryItemPresenter | None, modeIndex: int) -> None:
        self.parentItem = parentItem
        self._presenter = presenter
        self.modeIndex = modeIndex
        self.childItems: list[ProbeTreeNode] = list()

    @classmethod
    def createRoot(cls) -> ProbeTreeNode:
        return cls(None, None, -1)

    def createChild(self, presenter: ProbeRepositoryItemPresenter) -> ProbeTreeNode:
        childItem = ProbeTreeNode(self, presenter, -1)

        for modeIndex in range(presenter.item.getNumberOfProbeModes()):
            grandChildItem = ProbeTreeNode(childItem, presenter, modeIndex)
            childItem.childItems.append(grandChildItem)

        self.childItems.append(childItem)
        return childItem

    def getName(self) -> str:
        if self._presenter is None:
            return str()

        return self._presenter.name

    def getDataType(self) -> str:
        if self._presenter is None:
            return str()

        return self._presenter.item.getDataType()

    def getWidthInPixels(self) -> int:
        if self._presenter is None:
            return 0

        return self._presenter.item.getExtentInPixels().width

    def getHeightInPixels(self) -> int:
        if self._presenter is None:
            return 0

        return self._presenter.item.getExtentInPixels().height

    def getSizeInBytes(self) -> float:
        if self._presenter is None:
            return 0

        return self._presenter.item.getSizeInBytes()

    def getArray(self) -> ProbeArrayType:
        if self._presenter is None:
            return numpy.zeros((0, 0, 0), dtype=complex)

        if self.modeIndex < 0:
            return self._presenter.item.getProbeModesFlattened()

        return self._presenter.item.getProbeMode(self.modeIndex)

    def getRelativePowerPercent(self) -> int:
        if self._presenter is None or self.modeIndex < 0:
            return -1

        relativePower = self._presenter.item.getProbeModeRelativePower(self.modeIndex)
        return int((100 * relativePower).to_integral_value())

    def row(self) -> int:
        if self.parentItem:
            return self.parentItem.childItems.index(self)

        return 0


class ProbeTreeModel(QAbstractItemModel):

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rootNode = ProbeTreeNode.createRoot()
        self._header = [
            'Name', 'Relative Power', 'Data Type', 'Width [px]', 'Height [px]', 'Size [MB]'
        ]

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
            value = QVariant(self._header[section])

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

            if role == Qt.DisplayRole:
                if node.modeIndex < 0:
                    if index.column() == 0:
                        value = QVariant(node.getName())
                    elif index.column() == 2:
                        value = QVariant(node.getDataType())
                    elif index.column() == 3:
                        value = QVariant(node.getWidthInPixels())
                    elif index.column() == 4:
                        value = QVariant(node.getHeightInPixels())
                    elif index.column() == 5:
                        value = QVariant(f'{node.getSizeInBytes() / (1024 * 1024):.2f}')
                elif index.column() == 0:
                    value = QVariant(f'Mode {node.modeIndex}')
            elif role == Qt.UserRole and index.column() == 1:
                value = QVariant(node.getRelativePowerPercent())

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.childItems)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
