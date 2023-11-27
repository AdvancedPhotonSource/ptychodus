from __future__ import annotations
from typing import overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant

from ...api.probe import ProbeArrayType
from ...model.probe import ProbeRepositoryItemPresenter


class ProbeTreeNode:

    def __init__(self, parentNode: ProbeTreeNode | None,
                 presenter: ProbeRepositoryItemPresenter | None, mode: int) -> None:
        self.parentNode = parentNode
        self.presenter = presenter
        self.mode = mode
        self.children: list[ProbeTreeNode] = list()

    @classmethod
    def createRoot(cls) -> ProbeTreeNode:
        return cls(None, None, -1)

    def populateModes(self) -> None:
        if self.presenter is None:
            return

        self.children.clear()

        for mode in range(self.presenter.item.getProbe().getNumberOfModes()):
            childNode = ProbeTreeNode(self, self.presenter, mode)
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

    def getInitializerName(self) -> str:
        if self.presenter is None:
            return str()

        return self.presenter.item.getInitializerSimpleName()

    def getDataType(self) -> str:
        if self.presenter is None:
            return str()

        return str(self.presenter.item.getProbe().getDataType())

    def getNumberOfModes(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getProbe().getNumberOfModes()

    def getWidthInPixels(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getProbe().getImageExtent().widthInPixels

    def getHeightInPixels(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getProbe().getImageExtent().heightInPixels

    def getSizeInBytes(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getProbe().getSizeInBytes()

    def getArray(self) -> ProbeArrayType:
        if self.presenter is None:
            return numpy.zeros((0, 0, 0), dtype=complex)
        elif self.mode < 0:
            return self.presenter.item.getProbe().getModesFlattened()

        return self.presenter.item.getProbe().getMode(self.mode)

    def getRelativePowerPercent(self) -> int:
        if self.presenter is None or self.mode < 0:
            return -1

        relativePower = self.presenter.item.getProbe().getModeRelativePower(self.mode)

        if numpy.isfinite(relativePower):
            return int(100. * relativePower)

        return 0

    def row(self) -> int:
        if self.parentNode:
            return self.parentNode.children.index(self)

        return 0


class ProbeTreeModel(QAbstractItemModel):

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rootNode = ProbeTreeNode.createRoot()
        self._header = [
            'Name', 'Relative Power', 'Initializer', 'Data Type', 'Width [px]', 'Height [px]',
            'Size [MB]'
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
        value = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            value = QVariant(self._header[section])

        return value

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
                if node.mode < 0:
                    if index.column() == 0:
                        value = QVariant(node.getName())
                    elif index.column() == 2:
                        value = QVariant(node.getInitializerName())
                    elif index.column() == 3:
                        value = QVariant(node.getDataType())
                    elif index.column() == 4:
                        value = QVariant(node.getWidthInPixels())
                    elif index.column() == 5:
                        value = QVariant(node.getHeightInPixels())
                    elif index.column() == 6:
                        value = QVariant(f'{node.getSizeInBytes() / (1024 * 1024):.2f}')
                elif index.column() == 0:
                    value = QVariant(f'Mode {node.mode}')
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
