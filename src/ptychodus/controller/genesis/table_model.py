from __future__ import annotations
from enum import Enum
from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject

from ...model.genesis import GenesisStatusRepository


class _Column(Enum):
    """Every column of GenesisStatusTableModel: its order and its header.

    The member value is the header and the position comes from declaration order,
    so a column cannot be added to the ordering and forgotten in the headers.
    Bare names in a ``match`` arm are capture patterns, not value patterns.
    """

    FACILITY = 'Facility'
    LABEL = 'Label'
    ACTION = 'Action'
    STATUS = 'Status'
    START_TIME = 'Start Time'
    COMPLETION_TIME = 'Completion Time'

    def __init__(self, header: str) -> None:
        self.header = header
        # Members already defined when this one is constructed, which is its position.
        self.index = len(self.__class__.__members__)

    @classmethod
    def from_index(cls, index: int) -> _Column:
        """The column at `index`.

        Raises:
            IndexError: for a column outside the table, which the model's own
                ``columnCount`` makes impossible and so would be a bug rather than input.
        """
        if index < 0:
            raise IndexError(f'Column index {index} is negative.')

        return list(cls)[index]


class GenesisStatusTableModel(QAbstractTableModel):
    def __init__(self, presenter: GenesisStatusRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._dt_format = '%Y-%m-%d %H:%M:%S'

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            match orientation:
                case Qt.Orientation.Horizontal:
                    return _Column.from_index(section).header
                case Qt.Orientation.Vertical:
                    return section

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            status = self._presenter[index.row()]

            match _Column.from_index(index.column()):
                case _Column.FACILITY:
                    return status.facility
                case _Column.LABEL:
                    return status.label
                case _Column.ACTION:
                    return status.action
                case _Column.STATUS:
                    return status.status
                case _Column.START_TIME:
                    return status.start_time.strftime(self._dt_format)
                case _Column.COMPLETION_TIME:
                    if status.completion_time is not None:
                        return status.completion_time.strftime(self._dt_format)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(_Column)
