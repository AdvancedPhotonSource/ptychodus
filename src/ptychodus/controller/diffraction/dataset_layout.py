from typing import Any, overload

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ptychodus.api.tree import SimpleTreeNode

from ...model.diffraction import AssembledDiffractionDataset, DiffractionDatasetObserver
from ...view.diffraction import DatasetFileLayoutDialog


class SimpleTreeModel(QAbstractItemModel):
    def __init__(self, root_node: SimpleTreeNode, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._root_node = root_node

    def set_root_node(self, root_node: SimpleTreeNode) -> None:
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
                parent_item = child_item.parent_item

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
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._root_node.data(section)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        return super().flags(index)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parent_item = parent.internalPointer() if parent.isValid() else self._root_node
            child_item = parent_item.child_items[row]

            if child_item:
                value = self.createIndex(row, column, child_item)

        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            node = index.internalPointer()
            return node.data(index.column())

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.column() > 0:
            return 0

        node = self._root_node

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.child_items)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        node = self._root_node

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.item_data)


class DatasetLayoutViewController(DiffractionDatasetObserver):
    def __init__(self, dataset: AssembledDiffractionDataset, tree_model: SimpleTreeModel) -> None:
        super().__init__()
        self._dataset = dataset
        self._tree_model = tree_model

    @classmethod
    def show_dialog(cls, dataset: AssembledDiffractionDataset, parent: QWidget) -> None:
        tree_model = SimpleTreeModel(dataset.get_layout())
        controller = cls(dataset, tree_model)
        dataset.add_observer(controller)

        dialog = DatasetFileLayoutDialog(parent)
        dialog.setWindowTitle('Dataset File Layout')
        dialog.tree_view.setModel(tree_model)

        controller._sync_model_to_view()
        dialog.open()

    def _sync_model_to_view(self) -> None:
        self._tree_model.set_root_node(self._dataset.get_layout())

    def handle_bad_pixels_changed(self, num_bad_pixels: int) -> None:
        pass

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self._sync_model_to_view()
