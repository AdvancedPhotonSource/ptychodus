from __future__ import annotations
import logging
from collections.abc import Callable, Iterable
from typing import Any, Final, cast, overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QAbstractListModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush, QFont

from ptychodus.api.constants import LengthUnit, format_bytes
from ptychodus.api.diffraction import DiffractionPattern
from ptychodus.api.geometry import ImageExtent, PixelGeometry

from ptychodus.model.diffraction import (
    AssembledDiffractionArray,
    AssembledDiffractionDataset,
    DiffractionDatasetObserver,
    DiffractionDatasetRepository,
    DiffractionDatasetRepositoryObserver,
    DiffractionDatasetState,
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
        parent_node = self.parent_node

        if parent_node is None:
            return 0

        try:
            return parent_node.child_nodes.index(self)
        except ValueError:
            # Detached by a subtree rebuild. Qt can still ask while it tears the old
            # rows down, and -1 reads back as an invalid index rather than a wrong row.
            return -1


class _DatasetTreeNode(_TreeNode):
    """A dataset row, owning one array node per frame group.

    ``_array_nodes`` is the canonical structure and always holds one node per
    assembled array. ``child_nodes`` is the presentation, which hides the group
    level for a dataset that has only one frame group, since that group's columns
    and mean pattern would repeat its parent's exactly. Frame nodes are parented to
    whichever node actually shows them, so row and parent lookups need no special
    case for the collapsed shape.
    """

    def __init__(self, parent_node: _TreeNode, dataset: AssembledDiffractionDataset) -> None:
        super().__init__(parent_node)
        self._dataset = dataset
        self._array_nodes: list[_ArrayTreeNode] = []

    def get_dataset(self) -> AssembledDiffractionDataset:
        return self._dataset

    def get_num_arrays(self) -> int:
        return len(self._array_nodes)

    def set_arrays_from_dataset(self) -> None:
        """Re-read the array list, discarding nodes that view a replaced buffer."""
        self._array_nodes = [_ArrayTreeNode(self, array) for array in self._dataset]

    def add_array_node(self, array_row: int, array: AssembledDiffractionArray) -> _ArrayTreeNode:
        array_node = _ArrayTreeNode(self, array)
        self._array_nodes.insert(array_row, array_node)
        return array_node

    def create_children(self) -> list[_TreeNode]:
        """Build the presented children without installing them.

        The caller installs them inside the begin/end insert pair Qt requires, which
        is why this returns the list rather than assigning ``child_nodes``.
        """
        if not self._array_nodes:
            return []

        if len(self._array_nodes) == 1:
            array_node = self._array_nodes[0]
            array_node.child_nodes = []
            return _create_frame_nodes(array_node, self)

        children: list[_TreeNode] = []

        for array_node in self._array_nodes:
            array_node.child_nodes = _create_frame_nodes(array_node, array_node)
            children.append(array_node)

        return children

    def frame_offset_of(self, array_node: _ArrayTreeNode) -> int:
        """Position of the array's first frame among the dataset's frames.

        Recomputed on demand rather than cached: arrays are bisect-inserted by array
        index, so one landing out of order shifts every later offset, and only the
        handful of rows Qt is painting ever ask.
        """
        offset = 0

        for node in self._array_nodes:
            if node is array_node:
                break

            offset += node.get_nframes()

        return offset

    def get_label(self) -> str:
        return self._dataset.get_name()

    def get_counts(self) -> int:
        return int(self._dataset.get_mean_total_counts())

    def get_nframes(self) -> int:
        # Summed over the array nodes rather than the presented children, so the
        # answer is the same in both shapes and costs one term per frame group.
        return sum(node.get_nframes() for node in self._array_nodes)

    def get_data(self) -> DiffractionPattern | None:
        return self._dataset.get_mean_pattern()

    def get_detector_extent(self) -> ImageExtent:
        return self._dataset.get_metadata().detector_extent

    def get_raw_pixel_geometry(self) -> PixelGeometry:
        return self._dataset.get_raw_pixel_geometry()

    def get_processed_pixel_geometry(self) -> PixelGeometry:
        return self._dataset.get_processed_pixel_geometry()

    def get_num_bad_pixels(self) -> int:
        return int(numpy.count_nonzero(self._dataset.get_bad_pixels()))

    def get_nbytes(self) -> int:
        # The dataset's own total, not the sum of its arrays: the arrays are views into
        # one patterns buffer, and the shared index array and bad-pixel mask belong to
        # no single array. This keeps the column summing to the panel's info label.
        return self._dataset.get_nbytes()


class _ArrayTreeNode(_TreeNode):
    """A frame group. Owned by its dataset node even when it is not presented."""

    def __init__(self, dataset_node: _DatasetTreeNode, array: AssembledDiffractionArray) -> None:
        super().__init__(dataset_node)
        self._dataset_node = dataset_node
        self._array = array
        # Snapshot both reductions: each gathers through the shared buffer's index mask,
        # and the Frames and Size columns ask on every repaint.
        self._num_patterns = array.get_num_patterns()
        self._nbytes = array.get_patterns().nbytes

    def get_array(self) -> AssembledDiffractionArray:
        return self._array

    def get_frame_offset(self) -> int:
        return self._dataset_node.frame_offset_of(self)

    def get_label(self) -> str:
        return self._array.get_label()

    def get_counts(self) -> int:
        return int(self._array.get_mean_total_counts())

    def get_nframes(self) -> int:
        return self._num_patterns

    def get_nbytes(self) -> int:
        return self._nbytes

    def get_data(self) -> DiffractionPattern:
        return self._array.get_mean_pattern()


class _FrameTreeNode(_TreeNode):
    def __init__(
        self,
        parent_node: _TreeNode,
        array_node: _ArrayTreeNode,
        frame_index: int,
    ) -> None:
        super().__init__(parent_node)
        self._array_node = array_node
        self._frame_index = frame_index

    def get_label(self) -> str:
        # Numbered across the whole dataset, so a frame keeps its name whether or not
        # its group level is shown, and "Frame 3" names one frame rather than one per
        # group.
        return f'Frame {self._array_node.get_frame_offset() + self._frame_index}'

    def get_counts(self) -> int:
        return int(self._array_node.get_array().get_total_counts(self._frame_index))

    def get_nframes(self) -> int:
        return 1

    def get_nbytes(self) -> int:
        return self._array_node.get_array().get_pattern(self._frame_index).nbytes

    def get_data(self) -> DiffractionPattern:
        return self._array_node.get_array().get_pattern(self._frame_index)


def _create_frame_nodes(array_node: _ArrayTreeNode, parent_node: _TreeNode) -> list[_TreeNode]:
    """One frame node per pattern, parented to whichever node presents them."""
    return [
        _FrameTreeNode(parent_node, array_node, frame_index)
        for frame_index in range(array_node.get_nframes())
    ]


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
            row = current.get_row()
            return row if row >= 0 else None
        current = current.parent_node
    return None


class DiffractionTreeModel(QAbstractItemModel):
    """Tree of root → dataset → frame group → frame.

    The frame-group level is omitted for a dataset that holds a single group, whose
    frames hang directly off the dataset row instead; see
    :meth:`_DatasetTreeNode.create_children`.
    """

    def __init__(
        self,
        repository: DiffractionDatasetRepository,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
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
        self._update_max_counts(iter(dataset))
        self._replace_dataset_children(dataset_row, _DatasetTreeNode.set_arrays_from_dataset)

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
            # clear(), reload() and import_assembled_patterns() all replace the array
            # list wholesale, so the existing subtree views a buffer that is gone.
            self._update_max_counts(iter(dataset))
            self._replace_dataset_children(dataset_row, _DatasetTreeNode.set_arrays_from_dataset)
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

        self._update_max_counts([array])

        if dataset_node.get_num_arrays() < 2:
            # Crossing 0→1 hides the group level and 1→2 reveals it, so the dataset's
            # children are replaced wholesale. Cheap here: at most two groups of frames.
            self._replace_dataset_children(
                dataset_row, lambda node: node.add_array_node(array_row, array)
            )
            return

        # Already grouped, so the new group is one more row among its siblings.
        dataset_index = self.index(dataset_row, 0)
        self.beginInsertRows(dataset_index, array_row, array_row)
        array_node = dataset_node.add_array_node(array_row, array)
        array_node.child_nodes = _create_frame_nodes(array_node, array_node)
        dataset_node.child_nodes.insert(array_row, array_node)
        self.endInsertRows()

        # Frames are numbered across the dataset, so the later groups renumber.
        self._refresh_frame_labels(dataset_index, array_row + 1)

    def _replace_dataset_children(
        self, dataset_row: int, mutate: Callable[[_DatasetTreeNode], Any]
    ) -> None:
        """Swap one dataset's whole subtree, announced as a removal then an insert.

        Qt has no way to express a reparent, so every change that moves frames between
        the dataset row and a group row — a reload, or an array landing that hides or
        reveals the group level — goes through here.
        """
        dataset_node = self._dataset_node(dataset_row)
        if dataset_node is None:
            return

        dataset_index = self.index(dataset_row, 0)
        num_children = len(dataset_node.child_nodes)

        if num_children > 0:
            self.beginRemoveRows(dataset_index, 0, num_children - 1)
            dataset_node.child_nodes = []
            self.endRemoveRows()

        mutate(dataset_node)
        children = dataset_node.create_children()

        if children:
            self.beginInsertRows(dataset_index, 0, len(children) - 1)
            dataset_node.child_nodes = children
            self.endInsertRows()

    def _update_max_counts(self, arrays: Iterable[AssembledDiffractionArray]) -> None:
        """Raise the scale the counts bars are drawn against, restyling every row."""
        max_counts = max((int(array.get_max_total_counts()) for array in arrays), default=0)

        if self._max_counts < max_counts:
            self._max_counts = max_counts
            self._rebroadcast_counts()

    def _refresh_frame_labels(self, dataset_index: QModelIndex, first_array_row: int) -> None:
        """Renumber the frames of the groups an insert shifted."""
        for array_row in range(first_array_row, self.rowCount(dataset_index)):
            array_index = self.index(array_row, 0, dataset_index)
            num_frames = self.rowCount(array_index)

            if num_frames > 0:
                top_left = self.index(0, _COL_NAME, array_index)
                bottom_right = self.index(num_frames - 1, _COL_NAME, array_index)
                self.dataChanged.emit(top_left, bottom_right)

    def refresh_array(self, dataset_row: int, array_row: int) -> None:
        dataset_node = self._dataset_node(dataset_row)
        if dataset_node is None:
            return

        dataset_index = self.index(dataset_row, 0)
        if not dataset_index.isValid():
            return

        if dataset_node.get_num_arrays() < 2:
            # Collapsed: the group has no row of its own, so refresh the frames it
            # contributed together with the dataset row that summarizes them.
            self.refresh_dataset(dataset_row)
            self._refresh_child_rows(dataset_index)
            return

        top_left = self.index(array_row, 0, dataset_index)
        bottom_right = self.index(array_row, self.columnCount() - 1, dataset_index)
        self.dataChanged.emit(top_left, bottom_right)
        self._refresh_child_rows(top_left)

    def _refresh_child_rows(self, parent: QModelIndex) -> None:
        """Emit dataChanged across every direct child row of parent."""
        num_rows = self.rowCount(parent)

        if num_rows > 0:
            top_left = self.index(0, 0, parent)
            bottom_right = self.index(num_rows - 1, self.columnCount() - 1, parent)
            self.dataChanged.emit(top_left, bottom_right)

    def refresh_dataset(self, dataset_row: int) -> None:
        dataset_index = self.index(dataset_row, 0)
        if not dataset_index.isValid():
            return
        bottom_right = self.index(dataset_row, self.columnCount() - 1)
        self.dataChanged.emit(dataset_index, bottom_right)

    def _rebroadcast_counts(self, parent: QModelIndex = QModelIndex()) -> None:
        """Re-emit the counts column at every level below parent.

        The bars are percentages of _max_counts, so a brighter array landing restyles
        every existing row. Only nodes that have children are descended into, to avoid
        minting a QModelIndex per frame on a dataset that holds tens of thousands.
        """
        node = parent.internalPointer() if parent.isValid() else self._root
        num_rows = len(node.child_nodes)

        if num_rows == 0:
            return

        top_left = self.index(0, _COL_COUNTS, parent)
        bottom_right = self.index(num_rows - 1, _COL_COUNTS, parent)
        self.dataChanged.emit(top_left, bottom_right)

        for row, child_node in enumerate(node.child_nodes):
            if child_node.child_nodes:
                self._rebroadcast_counts(self.index(row, 0, parent))

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
                return f'{LengthUnit.MICROMETER.convert(node.get_raw_pixel_geometry().width_m):.4g}'
            case 6:
                return (
                    f'{LengthUnit.MICROMETER.convert(node.get_raw_pixel_geometry().height_m):.4g}'
                )
            case 7:
                return f'{LengthUnit.MICROMETER.convert(node.get_processed_pixel_geometry().width_m):.4g}'
            case 8:
                return f'{LengthUnit.MICROMETER.convert(node.get_processed_pixel_geometry().height_m):.4g}'
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

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if self.hasIndex(row, column, parent):
            parent_node = parent.internalPointer() if parent.isValid() else self._root
            child_node = parent_node.child_nodes[row]

            if child_node:
                return self.createIndex(row, column, child_node)

        return QModelIndex()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid() and parent.column() > 0:
            # Children hang off column 0 alone. Answering for every column would make
            # index(row, 0, parent) collide across the columns of one parent row.
            return 0

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
