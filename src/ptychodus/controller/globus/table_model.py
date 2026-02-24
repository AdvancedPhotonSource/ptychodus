from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QUrl
from PyQt5.QtGui import QColor, QFont

from ...model.globus import GlobusStatusRepository


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
                    case 0:
                        return flow_run.label
                    case 1:
                        return flow_run.start_time.strftime(self._dt_format)
                    case 2:
                        if flow_run.completion_time is not None:
                            return flow_run.completion_time.strftime(self._dt_format)
                    case 3:
                        return flow_run.status
                    case 4:
                        return flow_run.action
                    case 5:
                        return flow_run.run_id
            elif index.column() == 5:
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
