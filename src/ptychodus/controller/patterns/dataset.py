from __future__ import annotations
from typing import Any, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject
from PyQt5.QtGui import QFont

from ptychodus.api.patterns import (
    DiffractionPatternArray,
    PatternDataType,
    PatternState,
    SimpleDiffractionPatternArray,
)


class DatasetTreeNode:
    def __init__(
        self,
        parent_node: DatasetTreeNode | None,
        array: DiffractionPatternArray,
        frame_index: int,
    ) -> None:
        self.parent_node = parent_node
        self._array = array
        self._frame_index = frame_index
        self.child_nodes: list[DatasetTreeNode] = list()

    @classmethod
    def createRoot(cls) -> DatasetTreeNode:
        return cls(None, SimpleDiffractionPatternArray.createNullInstance(), -1)

    def createChild(self, array: DiffractionPatternArray) -> DatasetTreeNode:
        childItem = DatasetTreeNode(self, array, -1)

        if array.getData() is not None:
            for frame_index in range(array.getData().shape[0]):
                grandChildItem = DatasetTreeNode(childItem, array, frame_index)
                childItem.child_nodes.append(grandChildItem)

        self.child_nodes.append(childItem)
        return childItem

    @property
    def label(self) -> str:
        if self._frame_index < 0:
            return self._array.getLabel()

        return f'Frame {self._frame_index}'

    @property
    def state(self) -> PatternState:
        return self._array.getState()

    @property
    def data(self) -> PatternDataType | None:
        if self._array.getData() is None:
            return None
        elif self._frame_index < 0:
            return self._array.getData().mean(axis=0)

        return self._array.getData()[self._frame_index]

    @property
    def numberOfFrames(self) -> int:
        if self._frame_index < 0:
            return len(self.child_nodes)

        return 1

    @property
    def sizeInBytes(self) -> int:
        if self._array.getData() is None:
            return 0
        elif self._frame_index < 0:
            return self._array.getData().nbytes

        return self._array.getData()[self._frame_index].nbytes

    def row(self) -> int:
        if self.parent_node:
            return self.parent_node.child_nodes.index(self)

        return 0


class DatasetTreeModel(QAbstractItemModel):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rootNode = DatasetTreeNode.createRoot()
        self._header = ['Label', 'Frames', 'Size [MB]']

    def setRootNode(self, rootNode: DatasetTreeNode) -> None:
        self.beginResetModel()
        self._rootNode = rootNode
        self.endResetModel()

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...

    @overload
    def parent(self) -> QObject: ...

    def parent(self, child: QModelIndex | None = None) -> QModelIndex | QObject:
        if child is None:
            return super().parent()
        else:
            value = QModelIndex()

            if child.isValid():
                childItem = child.internalPointer()
                parent_node = childItem.parent_node

                if parent_node is self._rootNode:
                    value = QModelIndex()
                else:
                    value = self.createIndex(parent_node.row(), 0, parent_node)

            return value

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            node = index.internalPointer()

            if node.state != PatternState.LOADED:
                value &= ~Qt.ItemFlag.ItemIsSelectable
                value &= ~Qt.ItemFlag.ItemIsEnabled

        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            node = index.internalPointer()

            if role == Qt.ItemDataRole.DisplayRole:
                match index.column():
                    case 0:
                        return node.label
                    case 1:
                        return node.numberOfFrames
                    case 2:
                        return f'{node.sizeInBytes / (1024 * 1024):.2f}'
            elif role == Qt.ItemDataRole.FontRole:
                font = QFont()
                font.setItalic(node.state == PatternState.LOADING)
                return font

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parent_node = parent.internalPointer() if parent.isValid() else self._rootNode
            childItem = parent_node.child_nodes[row]

            if childItem:
                value = self.createIndex(row, column, childItem)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.child_nodes)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
