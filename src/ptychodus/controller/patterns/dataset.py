from __future__ import annotations
from typing import Any, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ptychodus.api.patterns import PatternDataType

from ptychodus.model.patterns import AssembledDiffractionPatternArray

__all__ = ['DatasetTreeModel']


class DatasetTreeNode:
    def __init__(
        self,
        parent_node: DatasetTreeNode | None,
        array: AssembledDiffractionPatternArray,
        frame_index: int,
    ) -> None:
        self.parent_node = parent_node
        self._array = array
        self._frame_index = frame_index
        self.child_nodes: list[DatasetTreeNode] = list()

    @classmethod
    def create_root(cls) -> DatasetTreeNode:
        return cls(None, AssembledDiffractionPatternArray.create_null(), -1)

    def insert_child(self, pos: int, array: AssembledDiffractionPatternArray) -> DatasetTreeNode:
        child = DatasetTreeNode(self, array, -1)

        for frame_index in range(array.getNumberOfPatterns()):
            grandchild = DatasetTreeNode(child, array, frame_index)
            child.child_nodes.append(grandchild)

        self.child_nodes.insert(pos, child)
        return child

    def get_label(self) -> str:
        return self._array.getLabel() if self._frame_index < 0 else f'Frame {self._frame_index}'

    def get_data(self) -> PatternDataType:
        return (
            self._array.get_average_pattern()
            if self._frame_index < 0
            else self._array.get_pattern(self._frame_index)
        )

    def get_counts(self) -> int:
        return (
            int(self._array.get_average_pattern_counts())
            if self._frame_index < 0
            else int(self._array.get_pattern_counts(self._frame_index))
        )

    def get_nframes(self) -> int:
        return len(self.child_nodes) if self._frame_index < 0 else 1

    def get_nbytes(self) -> int:
        return (
            self._array.getData().nbytes
            if self._frame_index < 0
            else self._array.get_pattern(self._frame_index).nbytes
        )

    def get_row(self) -> int:
        return 0 if self.parent_node is None else self.parent_node.child_nodes.index(self)


class DatasetTreeModel(QAbstractItemModel):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._nodes = DatasetTreeNode.create_root()
        self._header = ['Label', 'Counts', 'Frames', 'Size [MB]']

    def clear(self) -> None:
        self.beginResetModel()
        self._nodes = DatasetTreeNode.create_root()
        self.endResetModel()

    def insert_array(self, row: int, array: AssembledDiffractionPatternArray) -> None:
        self.beginInsertRows(QModelIndex(), row, row)
        child_node = self._nodes.insert_child(row, array)
        self.endInsertRows()

        index = self.index(row, 0)
        self.beginInsertRows(index, 0, len(child_node.child_nodes))
        self.endInsertRows()

    def refresh_array(self, row: int) -> None:
        topLeft = self.index(row, 0)
        bottomRight = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(topLeft, bottomRight)

        childTopLeft = self.index(row, 0, topLeft)
        childBottomRight = self.index(row, self.columnCount() - 1)
        self.dataChanged.emit(childTopLeft, childBottomRight)

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...

    @overload
    def parent(self) -> QObject: ...

    def parent(self, child: QModelIndex | None = None) -> QModelIndex | QObject:
        if child is None:
            return super().parent()

        if child.isValid():
            child_node = child.internalPointer()
            parent_node = child_node.parent_node

            if parent_node is not self._nodes:
                return self.createIndex(parent_node.get_row(), 0, parent_node)

        return QModelIndex()

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            node = index.internalPointer()

            if role == Qt.ItemDataRole.DisplayRole:
                match index.column():
                    case 0:
                        return node.get_label()
                    case 1:
                        return node.get_counts()
                    case 2:
                        return node.get_nframes()
                    case 3:
                        return f'{node.get_nbytes() / (1024 * 1024):.2f}'

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if self.hasIndex(row, column, parent):
            parent_node = parent.internalPointer() if parent.isValid() else self._nodes
            child_node = parent_node.child_nodes[row]

            if child_node:
                return self.createIndex(row, column, child_node)

        return QModelIndex()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = parent.internalPointer() if parent.isValid() else self._nodes
        return len(node.child_nodes)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
