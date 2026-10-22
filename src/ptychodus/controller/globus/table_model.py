from __future__ import annotations
from enum import Enum
from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QUrl
from PyQt5.QtGui import QColor, QFont

from ...model.globus import GlobusStatusRepository


class Column(Enum):
    """Every column of GlobusStatusTableModel: its order and its header.

    The member value is the header and the position comes from declaration order,
    so a column cannot be added to the ordering and forgotten in the headers.
    Public because the controller's click handler opens the run URL from one of them.
    """

    LABEL = 'Label'
    START_TIME = 'Start Time'
    COMPLETION_TIME = 'Completion Time'
    STATUS = 'Status'
    ACTION = 'Action'
    RUN_ID = 'Run ID'

    def __init__(self, header: str) -> None:
        self.header = header
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


class GlobusStatusTableModel(QAbstractTableModel):
    def __init__(self, presenter: GlobusStatusRepository, parent: QObject | None = None) -> None:
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
                    return Column.from_index(section).header
                case Qt.Orientation.Vertical:
                    return section

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            flow_run = self._presenter[index.row()]
            column = Column.from_index(index.column())

            if role == Qt.ItemDataRole.DisplayRole:
                match column:
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
            elif column is Column.RUN_ID:
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
        return len(Column)
