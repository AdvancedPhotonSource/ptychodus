from __future__ import annotations
from enum import Enum
from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush

from ptychodus.api.constants import format_bytes

from ...model.product import ProbePositionsAPI, ProbePositionsRepository
from ...model.product.probe_positions import ProbePositionsRepositoryItem


class Column(Enum):
    """Every column of ProbePositionsTableModel: its order, its header and whether it edits.

    The member value pairs the header with editability, and the position comes
    from declaration order rather than being written out, so a column cannot be added
    to the ordering and forgotten in the labels or the editable set. Nothing could pin
    that alignment: CLAUDE.md puts this file outside the tested surface.

    Public because the controller installs the builder combo delegate on one of them.
    """

    NAME = ('Name', True)
    PLOT = ('Plot', False)
    BUILDER = ('Builder', True)
    POINTS = ('Points', False)
    LENGTH_M = ('Length [m]', False)
    SIZE = ('Size', False)

    def __init__(self, header: str, editable: bool) -> None:
        self.header = header
        self.editable = editable
        # Members already defined when this one is constructed, which is its position.
        self.index = len(self.__class__.__members__)

    @classmethod
    def from_index(cls, index: int) -> Column:
        """The column at `index`.

        Raises:
            IndexError: for a column outside the table, which the model's own
                ``columnCount`` makes impossible and so would be a bug rather than input.
        """
        if index < 0:
            raise IndexError(f'Column index {index} is negative.')

        return list(cls)[index]


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
        self._checked_item_indexes: set[int] = set()

    def insert_item(self, index: int, item: ProbePositionsRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self.endInsertRows()

    def update_item(self, index: int, item: ProbePositionsRepositoryItem) -> None:
        # Qt row/column bounds are inclusive, so the last valid column is
        # len(Column) - 1. Asking index() for one past the end fails
        # hasIndex and yields an invalid QModelIndex, which makes the
        # dataChanged range malformed and the signal a no-op.
        last_column = len(Column) - 1
        top_left = self.index(index, 0)
        bottom_right = self.index(index, last_column)
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
            return Column.from_index(section).header

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        item = self._repository[index.row()]
        probe_positions = item.get_probe_positions()
        geometry = item.get_geometry()
        length_m = 0.0 if geometry is None else geometry.length_m

        column = Column.from_index(index.column())

        if role == Qt.ItemDataRole.DisplayRole:
            if column is Column.NAME:
                return self._repository.get_name(index.row())
            elif column is Column.PLOT:
                return None
            elif column is Column.BUILDER:
                return item.get_builder().get_name()
            elif column is Column.POINTS:
                return len(probe_positions)
            elif column is Column.LENGTH_M:
                return f'{length_m:.6f}'
            elif column is Column.SIZE:
                return format_bytes(probe_positions.nbytes)
        elif role == Qt.ItemDataRole.UserRole:
            if column is Column.SIZE:
                # Numeric sort key: the display text mixes units and cannot be
                # ordered as a string.
                return probe_positions.nbytes
        elif role == Qt.ItemDataRole.CheckStateRole:
            if column is Column.PLOT:
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
            column = Column.from_index(index.column())

            if column.editable:
                value |= Qt.ItemFlag.ItemIsEditable

            # Checkability belongs to the one column that has it, rather than to a
            # second tuple field every other column would carry as False.
            if column is Column.PLOT:
                value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
            return False

        column = Column.from_index(index.column())

        if role == Qt.ItemDataRole.EditRole:
            if column is Column.NAME:
                self._repository.set_name(index.row(), str(value))
                return True
            elif column is Column.BUILDER:
                self._api.build_probe_positions(index.row(), str(value))
                return True
        elif role == Qt.ItemDataRole.CheckStateRole:
            if column is Column.PLOT:
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
        return len(Column)
