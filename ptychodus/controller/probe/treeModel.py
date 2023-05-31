from __future__ import annotations
from typing import overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant

from ...api.probe import ProbeArrayType
from ...model.probe import ProbeRepositoryItemPresenter


class ProbeTreeNode:

    def __init__(self, parentNode: ProbeTreeNode | None,
                 presenter: ProbeRepositoryItemPresenter | None, probeMode: int) -> None:
        self.parentNode = parentNode
        self.presenter = presenter
        self.probeMode = probeMode
        self.children: list[ProbeTreeNode] = list()

    @classmethod
    def createRoot(cls) -> ProbeTreeNode:
        return cls(None, None, -1)

    def populateModes(self) -> None:
        if self.presenter is None:
            return

        self.children.clear()

        for probeMode in range(self.presenter.item.getNumberOfModes()):
            childNode = ProbeTreeNode(self, self.presenter, probeMode)
            self.children.append(childNode)

    def createChild(self, presenter: ProbeRepositoryItemPresenter) -> ProbeTreeNode:
        childNode = ProbeTreeNode(self, presenter, -1)
        childNode.populateModes()
        self.children.append(childNode)
        return childNode

    def getName(self) -> str:
        if self.presenter is None:
            return str()

        return self.presenter.name

    def getDataType(self) -> str:
        if self.presenter is None:
            return str()

        return self.presenter.item.getDataType()

    def getNumberOfModes(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getNumberOfModes()

    def getWidthInPixels(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getExtentInPixels().width

    def getHeightInPixels(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getExtentInPixels().height

    def getSizeInBytes(self) -> float:
        if self.presenter is None:
            return 0

        return self.presenter.item.getSizeInBytes()

    def getArray(self) -> ProbeArrayType:
        if self.presenter is None:
            return numpy.zeros((0, 0, 0), dtype=complex)
        elif self.probeMode < 0:
            return self.presenter.item.getModesFlattened()

        return self.presenter.item.getMode(self.probeMode)

    def getRelativePowerPercent(self) -> int:
        if self.presenter is None or self.probeMode < 0:
            return -1

        relativePower = self.presenter.item.getModeRelativePower(self.probeMode)
        return int((100 * relativePower).to_integral_value())

    def row(self) -> int:
        if self.parentNode:
            return self.parentNode.children.index(self)

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

    def refreshProbe(self, row: int) -> None:
        topLeft = self.index(row, 0)
        bottomRight = self.index(row, len(self._header))
        self.dataChanged.emit(topLeft, bottomRight)

        node = self._rootNode.children[row]
        numModesOld = len(node.children)
        numModesNew = node.getNumberOfModes()

        if numModesOld < numModesNew:
            self.beginInsertRows(topLeft, numModesOld, numModesNew)
            node.populateModes()
            self.endInsertRows()
        else:
            self.beginRemoveRows(topLeft, numModesNew, numModesOld)
            node.populateModes()
            self.endRemoveRows()

        childTopLeft = self.index(0, 0, topLeft)
        childBottomRight = self.index(numModesNew, len(self._header), topLeft)
        self.dataChanged.emit(childTopLeft, childBottomRight)

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
                childNode = child.internalPointer()
                parentNode = childNode.parentNode

                if parentNode is self._rootNode:
                    value = QModelIndex()
                else:
                    value = self.createIndex(parentNode.row(), 0, parentNode)

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
            parentNode = parent.internalPointer() if parent.isValid() else self._rootNode
            childNode = parentNode.children[row]

            if childNode:
                value = self.createIndex(row, column, childNode)

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            node = index.internalPointer()

            if role == Qt.DisplayRole:
                if node.probeMode < 0:
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
                    value = QVariant(f'Mode {node.probeMode}')
            elif role == Qt.UserRole and index.column() == 1:
                value = QVariant(node.getRelativePowerPercent())

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
