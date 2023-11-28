from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QVariant

from ...model.scan import ScanRepositoryPresenter


class ScanTableModel(QAbstractTableModel):

    def __init__(self, presenter: ScanRepositoryPresenter, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._header = ['Name', 'Initializer', 'Points', 'Length [m]', 'Size [MB]']
        self._checkedNames: set[str] = set()

    def isChecked(self, name: str) -> bool:
        return (name in self._checkedNames)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() == 0:
            value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            itemPresenter = self._presenter[index.row()]
            item = itemPresenter.item

            if role == Qt.ItemDataRole.DisplayRole:
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
            elif role == Qt.ItemDataRole.CheckStateRole:
                if index.column() == 0:
                    value = QVariant(Qt.CheckState.Checked if itemPresenter.name in
                                     self._checkedNames else Qt.CheckState.Unchecked)

        return value

    def setData(self,
                index: QModelIndex,
                value: QVariant,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            item = self._presenter[index.row()]

            if value == QVariant(Qt.CheckState.Checked):
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
