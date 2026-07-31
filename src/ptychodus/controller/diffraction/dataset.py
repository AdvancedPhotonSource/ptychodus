from __future__ import annotations
from typing import Any, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ptychodus.api.common import BYTES_PER_MEGABYTE
from ptychodus.api.diffraction import DiffractionPattern

from ptychodus.model.diffraction import AssembledDiffractionArray, AssembledDiffractionDataset

__all__ = ['DatasetTreeModel']


class _TreeNode:
    """Base tree node — root, dataset, array, or frame."""

    def __init__(self, parent_node: _TreeNode | None) -> None:
        self.parent_node = parent_node
        self.child_nodes: list[_TreeNode] = []

    def get_label(self) -> str:
        return ''

    def get_counts(self) -> int:
        return 0

    def get_nframes(self) -> int:
        return sum(child.get_nframes() for child in self.child_nodes)

    def get_nbytes(self) -> int:
        return sum(child.get_nbytes() for child in self.child_nodes)

    def get_data(self) -> DiffractionPattern | None:
        return None

    def get_row(self) -> int:
        return 0 if self.parent_node is None else self.parent_node.child_nodes.index(self)


class _DatasetTreeNode(_TreeNode):
    def __init__(self, parent_node: _TreeNode, dataset: AssembledDiffractionDataset) -> None:
        super().__init__(parent_node)
        self._dataset = dataset

    def get_dataset(self) -> AssembledDiffractionDataset:
        return self._dataset

    def get_label(self) -> str:
        return self._dataset.get_name()

    def get_counts(self) -> int:
        if not self.child_nodes:
            return 0
        return sum(child.get_counts() for child in self.child_nodes) // len(self.child_nodes)


class _ArrayTreeNode(_TreeNode):
    def __init__(self, parent_node: _TreeNode, array: AssembledDiffractionArray) -> None:
        super().__init__(parent_node)
        self._array = array
        for frame_index in range(array.get_num_patterns()):
            self.child_nodes.append(_FrameTreeNode(self, array, frame_index))

    def get_label(self) -> str:
        return self._array.get_label()

    def get_counts(self) -> int:
        return int(self._array.get_mean_pattern_counts())

    def get_max_counts(self) -> int:
        return int(self._array.get_max_pattern_counts())

    def get_nframes(self) -> int:
        return len(self.child_nodes)

    def get_nbytes(self) -> int:
        return self._array.get_patterns().nbytes

    def get_data(self) -> DiffractionPattern:
        return self._array.get_average_pattern()


class _FrameTreeNode(_TreeNode):
    def __init__(
        self,
        parent_node: _TreeNode,
        array: AssembledDiffractionArray,
        frame_index: int,
    ) -> None:
        super().__init__(parent_node)
        self._array = array
        self._frame_index = frame_index

    def get_label(self) -> str:
        return f'Frame {self._frame_index}'

    def get_counts(self) -> int:
        return int(self._array.get_pattern_counts(self._frame_index))

    def get_nframes(self) -> int:
        return 1

    def get_nbytes(self) -> int:
        return self._array.get_pattern(self._frame_index).nbytes

    def get_data(self) -> DiffractionPattern:
        return self._array.get_pattern(self._frame_index)


def _find_containing_dataset_row(node: _TreeNode) -> int | None:
    """Walk up until the dataset node is found; return its row within the root. None for the root."""
    current: _TreeNode | None = node
    while current is not None:
        if isinstance(current, _DatasetTreeNode):
            return current.get_row()
        current = current.parent_node
    return None


class DatasetTreeModel(QAbstractItemModel):
    """Three-level tree: root → dataset → array → frame."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root = _TreeNode(None)
        self._max_counts = 1
        self._header = ['Label', 'Counts', 'Frames', 'Size [MB]']

    def clear(self) -> None:
        self.beginResetModel()
        self._root = _TreeNode(None)
        self._max_counts = 1
        self.endResetModel()

    def insert_dataset(self, row: int, dataset: AssembledDiffractionDataset) -> None:
        self.beginInsertRows(QModelIndex(), row, row)
        dataset_node = _DatasetTreeNode(self._root, dataset)
        self._root.child_nodes.insert(row, dataset_node)
        self.endInsertRows()

    def remove_dataset(self, row: int) -> None:
        if not 0 <= row < len(self._root.child_nodes):
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        del self._root.child_nodes[row]
        self.endRemoveRows()

    def _dataset_node(self, dataset_row: int) -> _DatasetTreeNode | None:
        if not 0 <= dataset_row < len(self._root.child_nodes):
            return None
        node = self._root.child_nodes[dataset_row]
        assert isinstance(node, _DatasetTreeNode)
        return node

    def insert_array(
        self, dataset_row: int, array_row: int, array: AssembledDiffractionArray
    ) -> None:
        dataset_node = self._dataset_node(dataset_row)
        if dataset_node is None:
            return

        max_counts = int(array.get_max_pattern_counts())
        if self._max_counts < max_counts:
            self._max_counts = max_counts
            self._rebroadcast_counts()

        dataset_index = self.index(dataset_row, 0)
        self.beginInsertRows(dataset_index, array_row, array_row)
        array_node = _ArrayTreeNode(dataset_node, array)
        dataset_node.child_nodes.insert(array_row, array_node)
        self.endInsertRows()

        # Also announce the frame grandchildren.
        array_index = self.index(array_row, 0, dataset_index)
        num_frames = len(array_node.child_nodes)
        if num_frames > 0:
            self.beginInsertRows(array_index, 0, num_frames - 1)
            self.endInsertRows()

    def refresh_array(self, dataset_row: int, array_row: int) -> None:
        dataset_index = self.index(dataset_row, 0)
        if not dataset_index.isValid():
            return

        top_left = self.index(array_row, 0, dataset_index)
        bottom_right = self.index(array_row, self.columnCount() - 1, dataset_index)
        self.dataChanged.emit(top_left, bottom_right)

        num_rows = self.rowCount(top_left)
        num_cols = self.columnCount(top_left)
        if num_rows > 0:
            child_top_left = self.index(0, 0, top_left)
            child_bottom_right = self.index(num_rows - 1, num_cols - 1, top_left)
            self.dataChanged.emit(child_top_left, child_bottom_right)

    def refresh_dataset(self, dataset_row: int) -> None:
        dataset_index = self.index(dataset_row, 0)
        if not dataset_index.isValid():
            return
        bottom_right = self.index(dataset_row, self.columnCount() - 1)
        self.dataChanged.emit(dataset_index, bottom_right)

    def _rebroadcast_counts(self) -> None:
        num_rows = self.rowCount()
        if num_rows == 0:
            return
        top_left = self.index(0, 1)
        bottom_right = self.index(num_rows - 1, 1)
        self.dataChanged.emit(top_left, bottom_right)

        for dataset_row in range(num_rows):
            dataset_index = self.index(dataset_row, 0)
            num_arrays = self.rowCount(dataset_index)
            if num_arrays == 0:
                continue
            array_top_left = self.index(0, 1, dataset_index)
            array_bottom_right = self.index(num_arrays - 1, 1, dataset_index)
            self.dataChanged.emit(array_top_left, array_bottom_right)

            for array_row in range(num_arrays):
                array_index = self.index(array_row, 0, dataset_index)
                num_frames = self.rowCount(array_index)
                if num_frames == 0:
                    continue
                frame_top_left = self.index(0, 1, array_index)
                frame_bottom_right = self.index(num_frames - 1, 1, array_index)
                self.dataChanged.emit(frame_top_left, frame_bottom_right)

    def dataset_row_for_index(self, index: QModelIndex) -> int | None:
        """Return the dataset row that contains the given tree index, or None."""
        if not index.isValid():
            return None
        node = index.internalPointer()
        return _find_containing_dataset_row(node)

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

            if parent_node is not None and parent_node is not self._root:
                return self.createIndex(parent_node.get_row(), 0, parent_node)

        return QModelIndex()

    def headerData(  # noqa: N802
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
                        return str(node.get_counts())
                    case 2:
                        return node.get_nframes()
                    case 3:
                        return f'{node.get_nbytes() / BYTES_PER_MEGABYTE:.2f}'
            elif role == Qt.ItemDataRole.UserRole:
                if index.column() == 1:
                    return int(100 * node.get_counts()) // int(self._max_counts)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if self.hasIndex(row, column, parent):
            parent_node = parent.internalPointer() if parent.isValid() else self._root
            child_node = parent_node.child_nodes[row]

            if child_node:
                return self.createIndex(row, column, child_node)

        return QModelIndex()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        node = parent.internalPointer() if parent.isValid() else self._root
        return len(node.child_nodes)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)
