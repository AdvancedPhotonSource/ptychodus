from collections.abc import Sequence
from typing import Optional

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QUrl, QVariant
from PyQt5.QtGui import QColor, QFont

from ...model.workflow import WorkflowStatusPresenter, WorkflowStatus


class WorkflowTableModel(QAbstractTableModel):

    def __init__(self,
                 presenter: WorkflowStatusPresenter,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._sectionHeaders = [
            'Label',
            'Start Time',
            'Completion Time',
            'Status',
            'Action',
            'Run ID',
        ]

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                value = QVariant(self._sectionHeaders[section])
            elif orientation == Qt.Vertical:
                value = QVariant(section)

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            flowRun = self._presenter[index.row()]

            if role == Qt.DisplayRole:
                if index.column() == 0:
                    value = QVariant(flowRun.label)
                elif index.column() == 1:
                    value = QVariant(flowRun.startTime)
                elif index.column() == 2:
                    value = QVariant(flowRun.completionTime)
                elif index.column() == 3:
                    value = QVariant(flowRun.status)
                elif index.column() == 4:
                    value = QVariant(flowRun.action)
                elif index.column() == 5:
                    value = QVariant(flowRun.runID)
            elif index.column() == 5:
                if role == Qt.ToolTipRole:
                    value = QVariant(flowRun.runURL)
                elif role == Qt.FontRole:
                    font = QFont()
                    font.setUnderline(True)
                    value = QVariant(font)
                elif role == Qt.ForegroundRole:
                    color = QColor(Qt.blue)
                    value = QVariant(color)
                elif role == Qt.UserRole:
                    value = QVariant(QUrl(flowRun.runURL))

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._sectionHeaders)
