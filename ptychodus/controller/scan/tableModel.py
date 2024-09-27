from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject

from ...model.product import ScanAPI, ScanRepository
from ...model.product.scan import ScanRepositoryItem


class ScanTableModel(QAbstractTableModel):
    def __init__(
        self, repository: ScanRepository, api: ScanAPI, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._api = api
        self._header = ['Name', 'Plot', 'Builder', 'Points', 'Length [m]', 'Size [MB]']
        self._checkedItemIndexes: set[int] = set()

    def insertItem(self, index: int, item: ScanRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self.endInsertRows()

    def updateItem(self, index: int, item: ScanRepositoryItem) -> None:
        topLeft = self.index(index, 0)
        bottomRight = self.index(index, len(self._header))
        self.dataChanged.emit(topLeft, bottomRight)

    def removeItem(self, index: int, item: ScanRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self.endRemoveRows()

    def isItemChecked(self, itemIndex: int) -> bool:
        return itemIndex in self._checkedItemIndexes

    def headerData(
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

        item = self._repository[index.row()]
        scan = item.getScan()

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return self._repository.getName(index.row())
            elif index.column() == 1:
                return None
            elif index.column() == 2:
                return item.getBuilder().getName()
            elif index.column() == 3:
                return len(scan)
            elif index.column() == 4:
                return f'{item.getLengthInMeters():.6f}'
            elif index.column() == 5:
                return f'{scan.sizeInBytes / (1024 * 1024):.2f}'
        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() == 1:
                return (
                    Qt.CheckState.Checked
                    if index.row() in self._checkedItemIndexes
                    else Qt.CheckState.Unchecked
                )

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            if index.column() in (0, 2):
                value |= Qt.ItemFlag.ItemIsEditable

            if index.column() == 1:
                value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.EditRole:
            if index.column() == 0:
                self._repository.setName(index.row(), str(value))
                return True
            elif index.column() == 2:
                self._api.buildScan(index.row(), str(value))
                return True
        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() == 1:
                if value == Qt.CheckState.Checked:
                    self._checkedItemIndexes.add(index.row())
                else:
                    self._checkedItemIndexes.discard(index.row())

                self.dataChanged.emit(index, index)

                return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._repository)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
