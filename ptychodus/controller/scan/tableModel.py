from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QFont

from ...model import ScanInitializer, ScanPresenter


@dataclass(frozen=True)
class ScanRepositoryEntry:
    name: str
    initializer: ScanInitializer


class ScanTableModel(QAbstractTableModel):

    def __init__(self, presenter: ScanPresenter, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._scanList: list[ScanRepositoryEntry] = list()
        self._checkedNames: set[str] = set()

    def refresh(self) -> None:
        self.beginResetModel()
        self._scanList = [
            ScanRepositoryEntry(name, initializer)
            for name, initializer in self._presenter.getScanRepositoryContents()
        ]
        self.endResetModel()

    def isChecked(self, name: str) -> bool:
        return (name in self._checkedNames)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            entry = self._scanList[index.row()]

            if index.column() == 0:
                value = int(value) | Qt.ItemIsUserCheckable

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                result = QVariant('Name')
            elif section == 1:
                result = QVariant('Category')
            elif section == 2:
                result = QVariant('Variant')
            elif section == 3:
                result = QVariant('Length')

        return result

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            entry = self._scanList[index.row()]

            if role == Qt.CheckStateRole:
                if index.column() == 0:
                    value = QVariant(Qt.Checked if entry.name in
                                     self._checkedNames else Qt.Unchecked)
            elif role == Qt.DisplayRole:
                if index.column() == 0:
                    value = QVariant(entry.name)
                elif index.column() == 1:
                    value = QVariant(entry.initializer.category)
                elif index.column() == 2:
                    value = QVariant(entry.initializer.variant)
                elif index.column() == 3:
                    value = QVariant(len(entry.initializer))
            elif role == Qt.FontRole:
                font = QFont()
                font.setBold(entry.name == self._presenter.getActiveScan())
                value = QVariant(font)

        return value

    def setData(self, index: QModelIndex, value: QVariant, role: int = Qt.EditRole) -> bool:
        if index.isValid() and index.column() == 0 and role == Qt.CheckStateRole:
            entry = self._scanList[index.row()]

            if value == Qt.Checked:
                self._checkedNames.add(entry.name)
            else:
                self._checkedNames.discard(entry.name)

            self.dataChanged.emit(index, index)

            return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._scanList)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 4
