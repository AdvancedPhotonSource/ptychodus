from __future__ import annotations
from typing import overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant

from ...api.object import ObjectArrayType
from ...model.object import ObjectRepositoryItemPresenter


class ObjectTreeNode:

    def __init__(self, parentNode: ObjectTreeNode | None,
                 presenter: ObjectRepositoryItemPresenter | None, layer: int) -> None:
        self.parentNode = parentNode
        self.presenter = presenter
        self.layer = layer
        self.children: list[ObjectTreeNode] = list()

    @classmethod
    def createRoot(cls) -> ObjectTreeNode:
        return cls(None, None, -1)

    def populateLayers(self) -> None:
        if self.presenter is None:
            return

        self.children.clear()

        for layer in range(self.presenter.item.getObject().getNumberOfLayers()):
            childNode = ObjectTreeNode(self, self.presenter, layer)
            self.children.append(childNode)

    def createChild(self, presenter: ObjectRepositoryItemPresenter) -> ObjectTreeNode:
        childNode = ObjectTreeNode(self, presenter, -1)
        childNode.populateLayers()
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

        return str(self.presenter.item.getObject().getDataType())

    def getNumberOfLayers(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getObject().getNumberOfLayers()

    def getWidthInPixels(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getObject().getImageExtent().widthInPixels

    def getHeightInPixels(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getObject().getImageExtent().heightInPixels

    def getSizeInBytes(self) -> int:
        if self.presenter is None:
            return 0

        return self.presenter.item.getObject().getSizeInBytes()

    def getArray(self) -> ObjectArrayType:
        if self.presenter is None:
            return numpy.zeros((0, 0, 0), dtype=complex)
        elif self.layer < 0:
            return self.presenter.item.getObject().getLayersFlattened()

        return self.presenter.item.getObject().getLayer(self.layer)

    def getLayerDistanceInMeters(self) -> float:
        if self.presenter is None or self.layer < 0:
            return 0.

        distanceInMeters = self.presenter.item.getObject().getLayerDistanceInMeters(self.layer)
        return float(distanceInMeters)

    def row(self) -> int:
        if self.parentNode:
            return self.parentNode.children.index(self)

        return 0


class ObjectTreeModel(QAbstractItemModel):

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rootNode = ObjectTreeNode.createRoot()
        self._header = [
            'Name', 'Distance [m]', 'Initializer', 'Data Type', 'Width [px]', 'Height [px]',
            'Size [MB]'
        ]

    def setRootNode(self, rootNode: ObjectTreeNode) -> None:
        self.beginResetModel()
        self._rootNode = rootNode
        self.endResetModel()

    def refreshObject(self, row: int) -> None:
        topLeft = self.index(row, 0)
        bottomRight = self.index(row, len(self._header))
        self.dataChanged.emit(topLeft, bottomRight)

        node = self._rootNode.children[row]
        numLayersOld = len(node.children)
        numLayersNew = node.getNumberOfLayers()

        if numLayersOld < numLayersNew:
            self.beginInsertRows(topLeft, numLayersOld, numLayersNew)
            node.populateLayers()
            self.endInsertRows()
        else:
            self.beginRemoveRows(topLeft, numLayersNew, numLayersOld)
            node.populateLayers()
            self.endRemoveRows()

        childTopLeft = self.index(0, 0, topLeft)
        childBottomRight = self.index(numLayersNew, len(self._header), topLeft)
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

        if index.isValid() and role == Qt.DisplayRole:
            node = index.internalPointer()

            if node.layer < 0:
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
                value = QVariant(f'Layer {node.layer}')
            elif index.column() == 1:
                value = QVariant(node.getLayerDistanceInMeters())

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
