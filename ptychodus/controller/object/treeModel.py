from __future__ import annotations
from typing import overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant

from ...model.object import ObjectRepositoryItem
from ...model.product import ObjectRepository


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

    def __init__(self, repository: ObjectRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository
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
                   role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return QVariant(self._header[section])

        return QVariant()

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

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        if not index.isValid():
            return QVariant()

        parent = index.parent()

        if parent.isValid():
            item = self._repository[parent.row()]

            if role == Qt.ItemDataRole.DisplayRole:
                if index.column() == 0:
                    return QVariant(f'Layer {index.row() + 1}')
                if index.column() == 1:
                    object_ = item.getObject()
                    distanceInMeters = object_.getLayerDistanceInMeters(index.row())
                    return QVariant(distanceInMeters)
        else:
            item = self._repository[index.row()]
            object_ = item.getObject()

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                if index.column() == 0:
                    return QVariant(self._repository.getName(index.row()))
                elif index.column() == 1:
                    return QVariant()
                elif index.column() == 2:
                    return QVariant(item.getBuilder().getName())
                elif index.column() == 3:
                    return QVariant(str(object_.dataType))
                elif index.column() == 4:
                    return QVariant(object_.widthInPixels)
                elif index.column() == 5:
                    return QVariant(object_.heightInPixels)
                elif index.column() == 6:
                    return QVariant(f'{object_.sizeInBytes / (1024 * 1024):.2f}')

        return QVariant()

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            parent = index.parent()

            if not parent.isValid() and index.column() in (0, 2):
                value |= Qt.ItemFlag.ItemIsEditable

        return value

    def setData(self,
                index: QModelIndex,
                value: QVariant,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            parent = index.parent()

            if not parent.isValid():
                if index.column() == 0:
                    self._repository.setName(index.row(), value.value())
                    return True
                elif index.column() == 2:
                    self._repository.setBuilderByName(index.row(), value.value())
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = parent.internalPointer() if parent.isValid() else self._treeRoot
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
