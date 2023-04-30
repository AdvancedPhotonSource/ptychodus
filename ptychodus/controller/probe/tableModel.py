from typing import Optional

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QVariant

from ...model.probe import ProbePresenter


class ProbeModesTableModel(QAbstractTableModel):

    def __init__(self, presenter: ProbePresenter, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section == 0:
                value = QVariant('Mode')
            elif section == 1:
                value = QVariant('Relative Power')

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            if role == Qt.DisplayRole and index.column() == 0:
                value = QVariant(index.row())
            if role == Qt.UserRole and index.column() == 1:
                power = self._presenter.getProbeModeRelativePower(index.row())
                powerPct = int((100 * power).to_integral_value())
                value = QVariant(powerPct)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.getNumberOfProbeModes()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2
