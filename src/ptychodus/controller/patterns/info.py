from typing import Any, overload

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ptychodus.api.tree import SimpleTreeNode

from ...model.patterns import AssembledDiffractionDataset, DiffractionDatasetObserver
from ...view.patterns import PatternsInfoDialog


class SimpleTreeModel(QAbstractItemModel):
    def __init__(self, rootNode: SimpleTreeNode, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._rootNode = rootNode

    def setRootNode(self, rootNode: SimpleTreeNode) -> None:
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
                parentItem = childItem.parentItem

                if parentItem is self._rootNode:
                    value = QModelIndex()
                else:
                    value = self.createIndex(parentItem.row(), 0, parentItem)

            return value

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._rootNode.data(section)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        return super().flags(index)

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parentItem = parent.internalPointer() if parent.isValid() else self._rootNode
            childItem = parentItem.child_items[row]

            if childItem:
                value = self.createIndex(row, column, childItem)

        return value

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            node = index.internalPointer()
            return node.data(index.column())

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.child_items)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.item_data)


class PatternsInfoViewController(DiffractionDatasetObserver):
    def __init__(self, dataset: AssembledDiffractionDataset, treeModel: SimpleTreeModel) -> None:
        super().__init__()
        self._dataset = dataset
        self._treeModel = treeModel

    @classmethod
    def show_info(cls, dataset: AssembledDiffractionDataset, parent: QWidget) -> None:
        treeModel = SimpleTreeModel(dataset.get_contents_tree())
        controller = cls(dataset, treeModel)
        dataset.add_observer(controller)

        dialog = PatternsInfoDialog(parent)
        dialog.setWindowTitle('Patterns Info')
        dialog.treeView.setModel(treeModel)

        controller._syncModelToView()
        dialog.open()

    def _syncModelToView(self) -> None:
        self._treeModel.setRootNode(self._dataset.get_contents_tree())

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self._syncModelToView()
