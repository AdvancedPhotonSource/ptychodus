from __future__ import annotations
from typing import overload

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QFont

from ...api.data import DiffractionPatternArrayType, DiffractionPatternState
from ...model.data import DiffractionPatternArrayPresenter


class DatasetTreeNode:

    def __init__(self, parentItem: DatasetTreeNode | None,
                 presenter: DiffractionPatternArrayPresenter, frameIndex: int) -> None:
        self.parentItem = parentItem
        self._presenter = presenter
        self._frameIndex = frameIndex
        self.childItems: list[DatasetTreeNode] = list()

    @classmethod
    def createRoot(cls) -> DatasetTreeNode:
        return cls(None, DiffractionPatternArrayPresenter.createNull(), -1)

    def createChild(self, presenter: DiffractionPatternArrayPresenter) -> DatasetTreeNode:
        childItem = DatasetTreeNode(self, presenter, -1)

        if presenter.data is not None:
            for frameIndex in range(presenter.data.shape[0]):
                grandChildItem = DatasetTreeNode(childItem, presenter, frameIndex)
                childItem.childItems.append(grandChildItem)

        self.childItems.append(childItem)
        return childItem

    @property
    def label(self) -> str:
        if self._frameIndex < 0:
            return self._presenter.label

        return f'Frame {self._frameIndex}'

    @property
    def state(self) -> DiffractionPatternState:
        return self._presenter.state

    @property
    def data(self) -> DiffractionPatternArrayType | None:
        if self._presenter.data is None:
            return None
        elif self._frameIndex < 0:
            return self._presenter.data.sum(axis=0)

        return self._presenter.data[self._frameIndex]

    @property
    def numberOfFrames(self) -> int:
        if self._frameIndex < 0:
            return len(self.childItems)

        return 1

    @property
    def sizeInBytes(self) -> float:
        if self._presenter.data is None:
            return 0
        elif self._frameIndex < 0:
            return self._presenter.data.nbytes

        return self._presenter.data[self._frameIndex].nbytes

    def row(self) -> int:
        if self.parentItem:
            return self.parentItem.childItems.index(self)

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
    def parent(self, child: QModelIndex) -> QModelIndex:
        ...

    @overload
    def parent(self) -> QObject:
        ...

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

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = Qt.ItemFlags()

        if index.isValid():
            node = index.internalPointer()
            value = super().flags(index)

            if node.state != DiffractionPatternState.LOADED:
                value &= ~Qt.ItemIsSelectable
                value &= ~Qt.ItemIsEnabled

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            node = index.internalPointer()

            if role == Qt.DisplayRole:
                if index.column() == 0:
                    value = QVariant(node.label)
                elif index.column() == 1:
                    value = QVariant(node.numberOfFrames)
                elif index.column() == 2:
                    value = QVariant(f'{node.sizeInBytes / (1024 * 1024):.2f}')
            elif role == Qt.FontRole:
                font = QFont()
                font.setItalic(node.state == DiffractionPatternState.FOUND)
                value = QVariant(font)

        return value

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        value = QModelIndex()

        if self.hasIndex(row, column, parent):
            parentItem = parent.internalPointer() if parent.isValid() else self._rootNode
            childItem = parentItem.childItems[row]

            if childItem:
                value = self.createIndex(row, column, childItem)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.column() > 0:
            return 0

        node = self._rootNode

        if parent.isValid():
            node = parent.internalPointer()

        return len(node.childItems)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
