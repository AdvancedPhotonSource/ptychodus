from __future__ import annotations
import logging
from typing import Any, overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject
from PyQt5.QtWidgets import QWidget

from ptychodus.api.diffraction import DiffractionDatasetLayoutNode

from ...model.diffraction import AssembledDiffractionDataset, DiffractionDatasetObserver
from ...view.diffraction import DatasetEditorDialog

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


class DatasetEditorViewController(DiffractionDatasetObserver):
    def __init__(
        self,
        dataset: AssembledDiffractionDataset,
        tree_model: SimpleTreeModel,
        dialog: DatasetEditorDialog,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._tree_model = tree_model
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

        controller = cls(dataset, tree_model, dialog)
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
