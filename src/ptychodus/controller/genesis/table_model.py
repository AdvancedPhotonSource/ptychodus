from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject

from ...model.genesis import GenesisStatusRepository


class GenesisStatusTableModel(QAbstractTableModel):
    def __init__(self, presenter: GenesisStatusRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._section_headers = [
            'Label',
            'Start Time',
            'Completion Time',
            'Status',
            'Action',
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
                case 0:
                    return status.label
                case 1:
                    return status.start_time.strftime(self._dt_format)
                case 2:
                    if status.completion_time is not None:
                        return status.completion_time.strftime(self._dt_format)
                case 3:
                    return status.status
                case 4:
                    return status.action

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._section_headers)
