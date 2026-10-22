from __future__ import annotations
from enum import IntEnum
from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject

from ...model.genesis import GenesisStatusRepository


class _Column(IntEnum):
    """Column order of GenesisStatusTableModel.

    An enum rather than module-level ints so that ``match`` arms are value patterns:
    a bare name would be a capture pattern that swallows every column.
    """

    FACILITY = 0
    LABEL = 1
    ACTION = 2
    STATUS = 3
    START_TIME = 4
    COMPLETION_TIME = 5


class GenesisStatusTableModel(QAbstractTableModel):
    def __init__(self, presenter: GenesisStatusRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._section_headers = [
            'Facility',
            'Label',
            'Action',
            'Status',
            'Start Time',
            'Completion Time',
        ]
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
                    return self._section_headers[section]
                case Qt.Orientation.Vertical:
                    return section

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            status = self._presenter[index.row()]

            match index.column():
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
        return len(self._section_headers)
