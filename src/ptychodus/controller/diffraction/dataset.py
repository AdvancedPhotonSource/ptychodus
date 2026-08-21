from __future__ import annotations
import logging
from typing import Any, Final, cast, overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QAbstractListModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush, QFont

from ptychodus.api.constants import ONE_MICRON_M, format_bytes
from ptychodus.api.diffraction import DiffractionPattern
from ptychodus.api.geometry import ImageExtent, PixelGeometry

from ptychodus.model.diffraction import (
    AssembledDiffractionArray,
    AssembledDiffractionDataset,
    DiffractionDatasetObserver,
    DiffractionDatasetRepository,
    DiffractionDatasetRepositoryObserver,
    DiffractionDatasetState,
    PatternSizer,
)

__all__ = [
    'UNBOUND_DATASET',
    'DiffractionDatasetComboModel',
    'DiffractionTreeModel',
]

logger = logging.getLogger(__name__)

UNBOUND_DATASET: Final[str] = '(unbound)'
"""Label for the sentinel entry meaning "no diffraction dataset"."""

_COL_NAME = 0
_COL_COUNTS = 1
_COL_FRAMES = 2
_COL_WIDTH_PX = 3
_COL_HEIGHT_PX = 4
_COL_PHYSICAL_PIXEL_WIDTH_UM = 5
_COL_PHYSICAL_PIXEL_HEIGHT_UM = 6
_COL_PROCESSED_PIXEL_WIDTH_UM = 7
_COL_PROCESSED_PIXEL_HEIGHT_UM = 8
_COL_NUM_BAD_PIXELS = 9
_COL_SIZE = 10


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

    def get_data(self) -> DiffractionPattern | None:
        return self._dataset.get_average_pattern()

    def get_detector_extent(self) -> ImageExtent:
        return self._dataset.get_metadata().detector_extent

    def get_raw_pixel_geometry(self) -> PixelGeometry:
        return self._dataset.get_raw_pixel_geometry()

    def get_processed_pixel_geometry(self, sizer: PatternSizer) -> PixelGeometry:
        return sizer.get_processed_pixel_geometry(self._dataset.get_raw_pixel_geometry())

    def get_num_bad_pixels(self) -> int:
        return int(numpy.count_nonzero(self._dataset.get_bad_pixels()))

    def get_nbytes(self) -> int:
        # The dataset's own total, not the sum of its arrays: the arrays are views into
        # one patterns buffer, and the shared index array and bad-pixel mask belong to
        # no single array. This keeps the column summing to the panel's info label.
        return self._dataset.get_nbytes()


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


def _state_font(state: DiffractionDatasetState) -> QFont | None:
    """Italic while loading, struck through on failure — matching the Products table."""
    if state is DiffractionDatasetState.READY:
        return None

    font = QFont()
    font.setItalic(state is DiffractionDatasetState.PENDING)
    font.setStrikeOut(state is DiffractionDatasetState.FAILED)
    return font


def _find_containing_dataset_row(node: _TreeNode) -> int | None:
    """Walk up until the dataset node is found; return its row within the root. None for the root."""
    current: _TreeNode | None = node
    while current is not None:
        if isinstance(current, _DatasetTreeNode):
            return current.get_row()
        current = current.parent_node
    return None


class DiffractionTreeModel(QAbstractItemModel):
    """Three-level tree: root → dataset → array → frame."""

    def __init__(
        self,
        sizer: PatternSizer,
        repository: DiffractionDatasetRepository,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._sizer = sizer
        self._repository = repository
        self._root = _TreeNode(None)
        self._max_counts = 1
        self._row_observers: dict[int, _DatasetRowObserver] = {}
        self._header = [
            'Name',
            'Counts',
            'Frames',
            'Width\n[px]',
            'Height\n[px]',
            'Physical Pixel\nWidth [µm]',
            'Physical Pixel\nHeight [µm]',
            'Processed Pixel\nWidth [µm]',
            'Processed Pixel\nHeight [µm]',
            'Num Bad\nPixels',
            'Size',
        ]

        for dataset_row, dataset in enumerate(repository):
            self._attach_dataset(dataset_row, dataset)

        # Duck-typed; see class docstring on the ABC / sip metaclass conflict.
        repository.add_observer(cast(DiffractionDatasetRepositoryObserver, self))

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

    def _attach_dataset(self, dataset_row: int, dataset: AssembledDiffractionDataset) -> None:
        """Build the dataset's rows and start watching it for row-level changes."""
        self.insert_dataset(dataset_row, dataset)

        for array_row in range(len(dataset)):
            self.insert_array(dataset_row, array_row, dataset[array_row])

        observer = _DatasetRowObserver(self, dataset)
        dataset.add_observer(observer)
        self._row_observers[id(dataset)] = observer

    def handle_dataset_inserted(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        self._attach_dataset(index, dataset)

    def handle_dataset_removed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        observer = self._row_observers.pop(id(dataset), None)

        if observer is not None:
            dataset.remove_observer(observer)

        self.remove_dataset(index)

    def handle_metadata_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        self.refresh_dataset(index)

    def handle_state_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        self.refresh_dataset(index)

    def _row_of(self, dataset: AssembledDiffractionDataset) -> int | None:
        try:
            return list(self._repository).index(dataset)
        except ValueError:
            return None

    def _on_array_inserted(self, dataset: AssembledDiffractionDataset, array_row: int) -> None:
        dataset_row = self._row_of(dataset)

        if dataset_row is not None:
            self.insert_array(dataset_row, array_row, dataset[array_row])

    def _on_array_changed(self, dataset: AssembledDiffractionDataset, array_row: int) -> None:
        dataset_row = self._row_of(dataset)

        if dataset_row is not None:
            self.refresh_array(dataset_row, array_row)

    def _on_dataset_refreshed(self, dataset: AssembledDiffractionDataset) -> None:
        dataset_row = self._row_of(dataset)

        if dataset_row is not None:
            self.refresh_dataset(dataset_row)

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
        if not index.isValid():
            return None

        node = index.internalPointer()
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            match column:
                case 0:
                    return node.get_label()
                case 1:
                    return str(node.get_counts())
                case 2:
                    return node.get_nframes()
                case 10:
                    return format_bytes(node.get_nbytes())
                case _:
                    return self._dataset_column_display(node, column)
        elif role == Qt.ItemDataRole.EditRole:
            if column == _COL_NAME and isinstance(node, _DatasetTreeNode):
                return node.get_label()
        elif role == Qt.ItemDataRole.UserRole:
            if column == _COL_COUNTS:
                return int(100 * node.get_counts()) // int(self._max_counts)
        elif role == Qt.ItemDataRole.FontRole:
            return _state_font(self._state_of(node))
        elif role == Qt.ItemDataRole.ForegroundRole:
            if self._state_of(node) is not DiffractionDatasetState.READY:
                return QBrush(Qt.GlobalColor.gray)
        elif role == Qt.ItemDataRole.ToolTipRole:
            match self._state_of(node):
                case DiffractionDatasetState.PENDING:
                    return 'Loading…'
                case DiffractionDatasetState.FAILED:
                    return 'Load failed'
                case DiffractionDatasetState.READY:
                    return None
        return None

    def _state_of(self, node: _TreeNode) -> DiffractionDatasetState:
        """Load state of the dataset a node belongs to.

        Array and frame rows inherit their dataset's state, so a whole subtree greys
        out together while its patterns are still streaming in.
        """
        dataset_row = _find_containing_dataset_row(node)

        if dataset_row is None:
            return DiffractionDatasetState.READY

        try:
            return self._repository[dataset_row].get_state()
        except IndexError:
            return DiffractionDatasetState.READY

    def _dataset_column_display(self, node: _TreeNode, column: int) -> Any:
        # Columns 3..9 apply only to dataset rows; array/frame nodes render blank.
        if not isinstance(node, _DatasetTreeNode):
            return None

        match column:
            case 3:
                return node.get_detector_extent().width_px
            case 4:
                return node.get_detector_extent().height_px
            case 5:
                return f'{node.get_raw_pixel_geometry().width_m / ONE_MICRON_M:.4g}'
            case 6:
                return f'{node.get_raw_pixel_geometry().height_m / ONE_MICRON_M:.4g}'
            case 7:
                return (
                    f'{node.get_processed_pixel_geometry(self._sizer).width_m / ONE_MICRON_M:.4g}'
                )
            case 8:
                return (
                    f'{node.get_processed_pixel_geometry(self._sizer).height_m / ONE_MICRON_M:.4g}'
                )
            case 9:
                return node.get_num_bad_pixels()
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:  # noqa: N802
        base = super().flags(index)
        if not index.isValid():
            return base
        if index.column() != _COL_NAME:
            return base
        node = index.internalPointer()
        if not isinstance(node, _DatasetTreeNode):
            return base
        return base | Qt.ItemFlag.ItemIsEditable

    def setData(  # noqa: N802
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        if index.column() != _COL_NAME:
            return False
        node = index.internalPointer()
        if not isinstance(node, _DatasetTreeNode):
            return False

        new_name = str(value).strip()
        if not new_name:
            return False

        dataset = node.get_dataset()
        if new_name == dataset.get_name():
            return False

        unique_name = self._repository.create_unique_name(new_name)
        dataset.set_name(unique_name)
        # set_name notifies nobody on its own, and this is the only rename path, so
        # announce it through the repository: dataset-name consumers elsewhere (the
        # combo model, the product editor) have no other way to learn about it.
        self._repository.handle_metadata_changed(dataset)
        return True

    def refresh_processed_columns(self) -> None:
        """Emit dataChanged for the processed pixel-size columns of every dataset row.

        Called when the PatternSizer notifies (binning / transpose toggled or bin
        size edited) — since the processed columns are derived from the sizer
        transforms applied to each dataset's raw geometry, they all need to refresh
        even though no dataset itself changed.
        """
        num_rows = self.rowCount()
        if num_rows == 0:
            return
        top_left = self.index(0, _COL_PROCESSED_PIXEL_WIDTH_UM)
        bottom_right = self.index(num_rows - 1, _COL_PROCESSED_PIXEL_HEIGHT_UM)
        self.dataChanged.emit(top_left, bottom_right)

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


class _DatasetRowObserver(DiffractionDatasetObserver):
    """Feeds one dataset's row-level changes into the tree model.

    Captures the dataset reference at registration, since the callbacks carry only an
    array index. Its lifetime is owned by DiffractionTreeModel, which attaches one per
    dataset on insert and detaches it on removal.
    """

    def __init__(
        self, tree_model: DiffractionTreeModel, dataset: AssembledDiffractionDataset
    ) -> None:
        super().__init__()
        self._tree_model = tree_model
        self._dataset = dataset

    def handle_array_inserted(self, index: int) -> None:
        self._tree_model._on_array_inserted(self._dataset, index)

    def handle_array_changed(self, index: int) -> None:
        self._tree_model._on_array_changed(self._dataset, index)

    def handle_dataset_reloaded(self) -> None:
        self._tree_model._on_dataset_refreshed(self._dataset)

    def handle_pixel_geometry_changed(self) -> None:
        self._tree_model._on_dataset_refreshed(self._dataset)


class DiffractionDatasetComboModel(QAbstractListModel):
    """Single-column list of dataset names for a QComboBox, with an optional unbound entry.

    Observes DiffractionDatasetRepository so combos stay live as datasets are inserted,
    removed, or renamed -- the guarantee ProductRepositoryComboProxyModel already gives
    product combos. Duck-typed against the observer ABC; inheriting it would clash with
    sip's wrappertype metaclass on QAbstractListModel.

    The model keeps its own ordered list rather than indexing the repository directly:
    the repository has already mutated by the time an observer callback runs, and Qt
    requires rowCount() to still report the pre-change value inside beginInsertRows /
    beginRemoveRows.
    """

    def __init__(
        self,
        repository: DiffractionDatasetRepository,
        *,
        unbound_label: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._unbound_label = unbound_label
        self._datasets: list[AssembledDiffractionDataset] = list(repository)
        repository.add_observer(cast(DiffractionDatasetRepositoryObserver, self))

    @property
    def _offset(self) -> int:
        """Rows occupied by the unbound sentinel ahead of the first real dataset."""
        return 0 if self._unbound_label is None else 1

    def dataset_at(self, row: int) -> AssembledDiffractionDataset | None:
        """Dataset shown in a row, or None for the unbound sentinel or an invalid row."""
        dataset_row = row - self._offset

        if 0 <= dataset_row < len(self._datasets):
            return self._datasets[dataset_row]

        return None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._datasets) + self._offset

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            dataset = self.dataset_at(index.row())
            # Read the name live so a rename only needs a dataChanged, not a rebuild.
            return self._unbound_label if dataset is None else dataset.get_name()

        return None

    def handle_dataset_inserted(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        row = index + self._offset
        self.beginInsertRows(QModelIndex(), row, row)
        self._datasets.insert(index, dataset)
        self.endInsertRows()

    def handle_dataset_removed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        if not 0 <= index < len(self._datasets):
            return

        row = index + self._offset
        self.beginRemoveRows(QModelIndex(), row, row)
        del self._datasets[index]
        self.endRemoveRows()

    def handle_metadata_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        model_index = self.index(index + self._offset, 0)

        if model_index.isValid():
            self.dataChanged.emit(model_index, model_index)

    def handle_state_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        pass
