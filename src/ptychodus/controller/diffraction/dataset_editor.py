from __future__ import annotations
import logging
import math
from typing import Any, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QAbstractTableModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QWidget

from ptychodus.api.constants import ONE_MICRON_M
from ptychodus.api.diffraction import DiffractionDatasetLayoutNode
from ptychodus.api.geometry import PixelGeometry

from ...model.diffraction import AssembledDiffractionDataset, DiffractionDatasetObserver
from ...view.diffraction import DatasetEditorDialog
from ..helpers import create_brush_for_editable_cell

logger = logging.getLogger(__name__)


class SimpleTreeModel(QAbstractItemModel):
    _HEADERS = ('Name', 'Type', 'Details')

    def __init__(
        self, root_node: DiffractionDatasetLayoutNode, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._root_node = root_node

    def set_root_node(self, root_node: DiffractionDatasetLayoutNode) -> None:
        self.beginResetModel()
        self._root_node = root_node
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
                child_item = child.internalPointer()
                parent_item = child_item.parent

                if parent_item is self._root_node:
                    value = QModelIndex()
                else:
                    value = self.createIndex(parent_item.row(), 0, parent_item)

            return value

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
            and 0 <= section < len(self._HEADERS)
        ):
            return self._HEADERS[section]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        return super().flags(index)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parent_item = parent.internalPointer() if parent.isValid() else self._root_node
            child_item = parent_item.children[row]

            if child_item:
                value = self.createIndex(row, column, child_item)

        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            node = index.internalPointer()
            column = index.column()
            if 0 <= column < len(self._HEADERS):
                return (node.name, node.dtype, node.details)[column]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.column() > 0:
            return 0

        node = self._root_node

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._HEADERS)


_ROW_PIXEL_WIDTH_UM = 0
_ROW_PIXEL_HEIGHT_UM = 1


class DatasetPropertyTableModel(QAbstractTableModel):
    """Two-row properties table over an AssembledDiffractionDataset: pixel width / height in µm.

    Setting either row calls set_pixel_geometry_override on the dataset. The dataset then
    notifies observers via handle_pixel_geometry_changed(); the editor controller listens
    and calls beginResetModel/endResetModel to refresh the displayed values.
    """

    def __init__(
        self,
        dataset: AssembledDiffractionDataset,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._editable_item_brush = editable_item_brush
        self._header = ['Property', 'Value']
        self._properties = [
            'Physical Pixel Width [µm]',
            'Physical Pixel Height [µm]',
        ]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() == 1:
            if not self._dataset.is_load_in_progress():
                value |= Qt.ItemFlag.ItemIsEditable

        return value

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

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            geometry = self._dataset.get_raw_pixel_geometry()
            match (index.column(), index.row()):
                case (0, row):
                    return self._properties[row]
                case (1, r) if r == _ROW_PIXEL_WIDTH_UM:
                    return f'{geometry.width_m / ONE_MICRON_M:.4g}'
                case (1, r) if r == _ROW_PIXEL_HEIGHT_UM:
                    return f'{geometry.height_m / ONE_MICRON_M:.4g}'
        elif role == Qt.ItemDataRole.BackgroundRole:
            if index.flags() & Qt.ItemFlag.ItemIsEditable:
                return self._editable_item_brush

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid() or index.column() != 1:
            return False

        try:
            new_um = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(new_um) or new_um <= 0.0:
            return False
        new_m = new_um * ONE_MICRON_M

        current = self._dataset.get_raw_pixel_geometry()
        if index.row() == _ROW_PIXEL_WIDTH_UM:
            new_geometry = PixelGeometry(width_m=new_m, height_m=current.height_m)
        elif index.row() == _ROW_PIXEL_HEIGHT_UM:
            new_geometry = PixelGeometry(width_m=current.width_m, height_m=new_m)
        else:
            return False

        self._dataset.set_pixel_geometry_override(new_geometry)
        return True

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)


class DatasetEditorViewController(DiffractionDatasetObserver):
    def __init__(
        self,
        dataset: AssembledDiffractionDataset,
        tree_model: SimpleTreeModel,
        property_model: DatasetPropertyTableModel,
        dialog: DatasetEditorDialog,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._tree_model = tree_model
        self._property_model = property_model
        self._dialog = dialog

    @classmethod
    def edit_dataset(cls, dataset: AssembledDiffractionDataset, parent: QWidget) -> None:
        dialog = DatasetEditorDialog(parent)
        dialog.setWindowTitle(f'Edit Dataset: {dataset.get_name()}')

        tree_model = SimpleTreeModel(dataset.get_layout())
        dialog.tree_view.setModel(tree_model)
        tree_header = dialog.tree_view.header()
        if tree_header is not None:
            tree_header.setSectionResizeMode(tree_header.ResizeMode.ResizeToContents)

        editable_item_brush = create_brush_for_editable_cell(dialog.table_view)
        property_model = DatasetPropertyTableModel(dataset, editable_item_brush)
        dialog.table_view.setModel(property_model)
        vertical_header = dialog.table_view.verticalHeader()
        if vertical_header is not None:
            vertical_header.hide()
        table_header = dialog.table_view.horizontalHeader()
        if table_header is not None:
            table_header.setSectionResizeMode(table_header.ResizeMode.ResizeToContents)
        dialog.table_view.resizeRowsToContents()

        controller = cls(dataset, tree_model, property_model, dialog)
        dataset.add_observer(controller)

        dialog.finished.connect(controller._finish)
        dialog.open()
        dialog.adjustSize()

    def _finish(self, result: int) -> None:
        self._dataset.remove_observer(self)

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self._tree_model.set_root_node(self._dataset.get_layout())
        self._property_model.beginResetModel()
        self._property_model.endResetModel()

    def handle_pixel_geometry_changed(self) -> None:
        self._property_model.beginResetModel()
        self._property_model.endResetModel()
