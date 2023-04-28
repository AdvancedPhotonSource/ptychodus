from typing import Optional

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QFont

from ...model.scan import ScanRepositoryPresenter


class ScanTableModel(QAbstractTableModel):

    def __init__(self,
                 presenter: ScanRepositoryPresenter,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._header = ['Name', 'Initializer', 'Points', 'Length [m]', 'Size [MB]']
        self._checkedNames: set[str] = set()

    def isChecked(self, name: str) -> bool:
        return (name in self._checkedNames)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() == 0:
            value = int(value) | Qt.ItemIsUserCheckable

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            itemPresenter = self._presenter[index.row()]
            item = itemPresenter.item

            if role == Qt.DisplayRole:
                if index.column() == 0:
                    value = QVariant(itemPresenter.name)
                elif index.column() == 1:
                    value = QVariant(item.getInitializerSimpleName())
                elif index.column() == 2:
                    value = QVariant(len(item))
                elif index.column() == 3:
                    value = QVariant(f'{item.getLengthInMeters():.6f}')
                elif index.column() == 4:
                    value = QVariant(f'{item.getSizeInBytes() / (1024 * 1024):.2f}')
            elif role == Qt.CheckStateRole:
                if index.column() == 0:
                    value = QVariant(Qt.Checked if itemPresenter.name in
                                     self._checkedNames else Qt.Unchecked)

        return value

    def setData(self, index: QModelIndex, value: QVariant, role: int = Qt.EditRole) -> bool:
        if index.isValid() and index.column() == 0 and role == Qt.CheckStateRole:
            item = self._presenter[index.row()]

            if value == QVariant(Qt.Checked):
                self._checkedNames.add(item.name)
            else:
                self._checkedNames.discard(item.name)

            self.dataChanged.emit(index, index)

            return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
