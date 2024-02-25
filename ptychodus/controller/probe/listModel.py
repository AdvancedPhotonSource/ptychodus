from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject

from ...model.probe import ProbeRepositoryItem
from ...model.product import ProbeRepository


class ProbeListModel(QAbstractListModel):

    def __init__(self, repository: ProbeRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository

    def insertItem(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self.endInsertRows()

    def updateItem(self, index: int, item: ProbeRepositoryItem) -> None:
        topLeft = self.index(index, 0)
        bottomRight = topLeft
        self.dataChanged.emit(topLeft, bottomRight)

    def removeItem(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self.endRemoveRows()

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            return self._repository.getName(index.row())

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._repository)
