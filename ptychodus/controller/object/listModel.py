from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QVariant

from ...model.object import ObjectRepositoryItem
from ...model.product import ObjectRepository


class ObjectListModel(QAbstractListModel):

    def __init__(self, repository: ObjectRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository

    def insertItem(self, index: int, item: ObjectRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self.endInsertRows()

    def updateItem(self, index: int, item: ObjectRepositoryItem) -> None:
        topLeft = self.index(index, 0)
        bottomRight = topLeft
        self.dataChanged.emit(topLeft, bottomRight)

    def removeItem(self, index: int, item: ObjectRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self.endRemoveRows()

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return QVariant(self._repository.getName(index.row()))

        return QVariant()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._repository)
