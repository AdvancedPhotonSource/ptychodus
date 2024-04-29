from __future__ import annotations
from typing import Any, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ...model.product import ObjectAPI, ObjectRepository
from ...model.product.object import ObjectRepositoryItem


class ObjectTreeNode:

    def __init__(self, parent: ObjectTreeNode | None = None) -> None:
        self.parent = parent
        self.children: list[ObjectTreeNode] = list()

    def insertNode(self, index: int = -1) -> ObjectTreeNode:
        node = ObjectTreeNode(self)
        self.children.insert(index, node)
        return node

    def removeNode(self, index: int = -1) -> ObjectTreeNode:
        return self.children.pop(index)

    def row(self) -> int:
        return 0 if self.parent is None else self.parent.children.index(self)


class ObjectTreeModel(QAbstractItemModel):

    def __init__(self,
                 repository: ObjectRepository,
                 api: ObjectAPI,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository
        self._api = api
        self._treeRoot = ObjectTreeNode()
        self._header = [
            'Name', 'Distance [m]', 'Builder', 'Data Type', 'Width [px]', 'Height [px]',
            'Size [MB]'
        ]

    @staticmethod
    def _appendLayers(node: ObjectTreeNode, item: ObjectRepositoryItem) -> None:
        object_ = item.getObject()

        for layer in range(object_.numberOfLayers):
            node.insertNode()

    def insertItem(self, index: int, item: ObjectRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        ObjectTreeModel._appendLayers(self._treeRoot.insertNode(index), item)
        self.endInsertRows()

    def updateItem(self, index: int, item: ObjectRepositoryItem) -> None:
        topLeft = self.index(index, 0)
        bottomRight = self.index(index, len(self._header))
        self.dataChanged.emit(topLeft, bottomRight)

        node = self._treeRoot.children[index]
        numLayersOld = len(node.children)
        numLayersNew = item.getObject().numberOfLayers

        if numLayersOld < numLayersNew:
            self.beginInsertRows(topLeft, numLayersOld, numLayersNew)

            while len(node.children) < numLayersNew:
                node.insertNode()

            self.endInsertRows()
        elif numLayersOld > numLayersNew:
            self.beginRemoveRows(topLeft, numLayersNew, numLayersOld)

            while len(node.children) > numLayersNew:
                node.removeNode()

            self.endRemoveRows()

        childTopLeft = self.index(0, 0, topLeft)
        childBottomRight = self.index(numLayersNew, len(self._header), topLeft)
        self.dataChanged.emit(childTopLeft, childBottomRight)

    def removeItem(self, index: int, item: ObjectRepositoryItem) -> None:
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

            if role == Qt.ItemDataRole.DisplayRole:
                if index.column() == 0:
                    return f'Layer {index.row() + 1}'
                elif index.column() == 1:
                    try:
                        return item.layerDistanceInMeters[index.row()]
                    except IndexError:
                        return float('NaN')
        else:
            item = self._repository[index.row()]
            object_ = item.getObject()

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                if index.column() == 0:
                    return self._repository.getName(index.row())
                elif index.column() == 1:
                    return object_.getTotalLayerDistanceInMeters()
                elif index.column() == 2:
                    return item.getBuilder().getName()
                elif index.column() == 3:
                    return str(object_.dataType)
                elif index.column() == 4:
                    return object_.widthInPixels
                elif index.column() == 5:
                    return object_.heightInPixels
                elif index.column() == 6:
                    return f'{object_.sizeInBytes / (1024 * 1024):.2f}'

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            parent = index.parent()

            if parent.isValid():
                if index.column() == 1:
                    item = self._repository[parent.row()]

                    if index.row() + 1 < item.getNumberOfLayers():
                        value |= Qt.ItemFlag.ItemIsEditable
            else:
                if index.column() in (0, 2):
                    value |= Qt.ItemFlag.ItemIsEditable

        return value

    def setData(self,
                index: QModelIndex,
                value: Any,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            parent = index.parent()

            if parent.isValid():
                item = self._repository[parent.row()]

                if index.column() == 1:
                    try:
                        distanceInM = float(value)
                    except ValueError:
                        return False

                    item.layerDistanceInMeters[index.row()] = distanceInM
                    return False
            else:
                if index.column() == 0:
                    self._repository.setName(index.row(), str(value))
                    return True
                elif index.column() == 2:
                    self._api.buildObject(index.row(), str(value))
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = parent.internalPointer() if parent.isValid() else self._treeRoot
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
