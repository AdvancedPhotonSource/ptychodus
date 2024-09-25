from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QUrl
from PyQt5.QtGui import QColor, QFont

from ...model.workflow import WorkflowStatusPresenter


class WorkflowTableModel(QAbstractTableModel):
    def __init__(self, presenter: WorkflowStatusPresenter, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._sectionHeaders = [
            "Label",
            "Start Time",
            "Completion Time",
            "Status",
            "Action",
            "Run ID",
        ]
        self._dtFormat = "%Y-%m-%d %H:%M:%S"

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self._sectionHeaders[section]
            elif orientation == Qt.Orientation.Vertical:
                return section

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            flowRun = self._presenter[index.row()]

            if role == Qt.ItemDataRole.DisplayRole:
                if index.column() == 0:
                    return flowRun.label
                elif index.column() == 1:
                    return flowRun.startTime.strftime(self._dtFormat)
                elif index.column() == 2:
                    if flowRun.completionTime is not None:
                        return flowRun.completionTime.strftime(self._dtFormat)
                elif index.column() == 3:
                    return flowRun.status
                elif index.column() == 4:
                    return flowRun.action
                elif index.column() == 5:
                    return flowRun.runID
            elif index.column() == 5:
                if role == Qt.ItemDataRole.ToolTipRole:
                    return flowRun.runURL
                elif role == Qt.ItemDataRole.FontRole:
                    font = QFont()
                    font.setUnderline(True)
                    return font
                elif role == Qt.ItemDataRole.ForegroundRole:
                    color = QColor(Qt.GlobalColor.blue)
                    return color
                elif role == Qt.ItemDataRole.UserRole:
                    return QUrl(flowRun.runURL)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._sectionHeaders)
