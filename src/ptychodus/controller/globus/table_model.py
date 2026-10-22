from __future__ import annotations
from enum import IntEnum
from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QUrl
from PyQt5.QtGui import QColor, QFont

from ...model.globus import GlobusStatusRepository


class Column(IntEnum):
    """Column order of GlobusStatusTableModel.

    An enum rather than module-level ints so that ``match`` arms are value patterns:
    a bare name would be a capture pattern that swallows every column. Public because
    the controller's click handler opens the run URL from one of these columns.
    """

    LABEL = 0
    START_TIME = 1
    COMPLETION_TIME = 2
    STATUS = 3
    ACTION = 4
    RUN_ID = 5


class GlobusStatusTableModel(QAbstractTableModel):
    def __init__(self, presenter: GlobusStatusRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._section_headers = [
            'Label',
            'Start Time',
            'Completion Time',
            'Status',
            'Action',
            'Run ID',
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
        if index.isValid():
            flow_run = self._presenter[index.row()]

            if role == Qt.ItemDataRole.DisplayRole:
                match index.column():
                    case Column.LABEL:
                        return flow_run.label
                    case Column.START_TIME:
                        return flow_run.start_time.strftime(self._dt_format)
                    case Column.COMPLETION_TIME:
                        if flow_run.completion_time is not None:
                            return flow_run.completion_time.strftime(self._dt_format)
                    case Column.STATUS:
                        return flow_run.status
                    case Column.ACTION:
                        return flow_run.action
                    case Column.RUN_ID:
                        return flow_run.run_id
            elif index.column() == Column.RUN_ID:
                match role:
                    case Qt.ItemDataRole.ToolTipRole:
                        return flow_run.run_url
                    case Qt.ItemDataRole.FontRole:
                        font = QFont()
                        font.setUnderline(True)
                        return font
                    case Qt.ItemDataRole.ForegroundRole:
                        color = QColor(Qt.GlobalColor.blue)
                        return color
                    case Qt.ItemDataRole.UserRole:
                        return QUrl(flow_run.run_url)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._section_headers)
