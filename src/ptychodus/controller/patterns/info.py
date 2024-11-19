from typing import Any, overload

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.tree import SimpleTreeNode

from ...model.patterns import DiffractionDatasetPresenter
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
            childItem = parentItem.childItems[row]

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

        return len(node.childItems)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.itemData)


class PatternsInfoViewController(Observer):
    def __init__(self, presenter: DiffractionDatasetPresenter, treeModel: SimpleTreeModel) -> None:
        super().__init__()
        self._presenter = presenter
        self._treeModel = treeModel

    @classmethod
    def showInfo(cls, presenter: DiffractionDatasetPresenter, parent: QWidget) -> None:
        treeModel = SimpleTreeModel(presenter.getContentsTree())
        controller = cls(presenter, treeModel)
        presenter.addObserver(controller)

        dialog = PatternsInfoDialog.createInstance(parent)
        dialog.setWindowTitle('Patterns Info')
        dialog.treeView.setModel(treeModel)
        dialog.finished.connect(controller._finish)

        controller._syncModelToView()
        dialog.open()

    def _finish(self, result: int) -> None:
        self._presenter.removeObserver(self)

    def _syncModelToView(self) -> None:
        self._treeModel.setRootNode(self._presenter.getContentsTree())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
