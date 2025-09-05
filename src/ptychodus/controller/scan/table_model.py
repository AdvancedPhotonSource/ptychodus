from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush

from ptychodus.api.units import BYTES_PER_MEGABYTE

from ...model.product import ScanAPI, ScanRepository
from ...model.product.scan import ScanRepositoryItem


class ScanTableModel(QAbstractTableModel):
    def __init__(
        self,
        repository: ScanRepository,
        api: ScanAPI,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._api = api
        self._editable_item_brush = editable_item_brush
        self._header = ['Name', 'Plot', 'Builder', 'Points', 'Length [m]', 'Size [MB]']
        self._checked_item_indexes: set[int] = set()

    def insert_item(self, index: int, item: ScanRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self.endInsertRows()

    def update_item(self, index: int, item: ScanRepositoryItem) -> None:
        top_left = self.index(index, 0)
        bottom_right = self.index(index, len(self._header))
        self.dataChanged.emit(top_left, bottom_right)

    def remove_item(self, index: int, item: ScanRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self.endRemoveRows()

    def is_item_checked(self, item_index: int) -> bool:
        return item_index in self._checked_item_indexes

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

        item = self._repository[index.row()]
        scan = item.get_scan()

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return self._repository.get_name(index.row())
            elif index.column() == 1:
                return None
            elif index.column() == 2:
                return item.get_builder().get_name()
            elif index.column() == 3:
                return len(scan)
            elif index.column() == 4:
                return f'{item.get_length_m():.6f}'
            elif index.column() == 5:
                return f'{scan.nbytes / BYTES_PER_MEGABYTE:.2f}'
        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() == 1:
                return (
                    Qt.CheckState.Checked
                    if index.row() in self._checked_item_indexes
                    else Qt.CheckState.Unchecked
                )
        elif role == Qt.ItemDataRole.BackgroundRole:
            if index.flags() & Qt.ItemFlag.ItemIsEditable:
                return self._editable_item_brush

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            if index.column() in (0, 2):
                value |= Qt.ItemFlag.ItemIsEditable

            if index.column() == 1:
                value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.EditRole:
            if index.column() == 0:
                self._repository.set_name(index.row(), str(value))
                return True
            elif index.column() == 2:
                self._api.build_scan(index.row(), str(value))
                return True
        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() == 1:
                if value == Qt.CheckState.Checked:
                    self._checked_item_indexes.add(index.row())
                else:
                    self._checked_item_indexes.discard(index.row())

                self.dataChanged.emit(index, index)

                return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._repository)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)
