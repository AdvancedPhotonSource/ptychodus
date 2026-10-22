from __future__ import annotations
from enum import IntEnum
from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush

from ptychodus.api.constants import format_bytes

from ...model.product import ProbePositionsAPI, ProbePositionsRepository
from ...model.product.probe_positions import ProbePositionsRepositoryItem


class Column(IntEnum):
    """Column order of ProbePositionsTableModel.

    An enum rather than module-level ints so that ``match`` arms are value patterns:
    a bare name would be a capture pattern that swallows every column. Public because
    the controller installs the builder combo delegate on one of these columns.
    """

    NAME = 0
    PLOT = 1
    BUILDER = 2
    POINTS = 3
    LENGTH_M = 4
    SIZE = 5


class ProbePositionsTableModel(QAbstractTableModel):
    def __init__(
        self,
        repository: ProbePositionsRepository,
        api: ProbePositionsAPI,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._api = api
        self._editable_item_brush = editable_item_brush
        self._header = ['Name', 'Plot', 'Builder', 'Points', 'Length [m]', 'Size']
        self._checked_item_indexes: set[int] = set()

    def insert_item(self, index: int, item: ProbePositionsRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self.endInsertRows()

    def update_item(self, index: int, item: ProbePositionsRepositoryItem) -> None:
        top_left = self.index(index, 0)
        bottom_right = self.index(index, len(self._header))
        self.dataChanged.emit(top_left, bottom_right)

    def remove_item(self, index: int, item: ProbePositionsRepositoryItem) -> None:
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
        probe_positions = item.get_probe_positions()
        geometry = item.get_geometry()
        length_m = 0.0 if geometry is None else geometry.length_m

        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == Column.NAME:
                return self._repository.get_name(index.row())
            elif index.column() == Column.PLOT:
                return None
            elif index.column() == Column.BUILDER:
                return item.get_builder().get_name()
            elif index.column() == Column.POINTS:
                return len(probe_positions)
            elif index.column() == Column.LENGTH_M:
                return f'{length_m:.6f}'
            elif index.column() == Column.SIZE:
                return format_bytes(probe_positions.nbytes)
        elif role == Qt.ItemDataRole.UserRole:
            if index.column() == Column.SIZE:
                # Numeric sort key: the display text mixes units and cannot be
                # ordered as a string.
                return probe_positions.nbytes
        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() == Column.PLOT:
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
            if index.column() in (Column.NAME, Column.BUILDER):
                value |= Qt.ItemFlag.ItemIsEditable

            if index.column() == Column.PLOT:
                value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.EditRole:
            if index.column() == Column.NAME:
                self._repository.set_name(index.row(), str(value))
                return True
            elif index.column() == Column.BUILDER:
                self._api.build_probe_positions(index.row(), str(value))
                return True
        elif role == Qt.ItemDataRole.CheckStateRole:
            if index.column() == Column.PLOT:
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
