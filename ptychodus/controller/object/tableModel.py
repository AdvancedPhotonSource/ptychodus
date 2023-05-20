from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QVariant

from ...model.object import ObjectRepositoryPresenter


class ObjectTableModel(QAbstractTableModel):

    def __init__(self,
                 presenter: ObjectRepositoryPresenter,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._header = ['Name', 'Data Type', 'Width [px]', 'Height [px]', 'Size [MB]']

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

        if index.isValid() and role == Qt.DisplayRole:
            itemPresenter = self._presenter[index.row()]
            item = itemPresenter.item

            if index.column() == 0:
                value = QVariant(itemPresenter.name)
            elif index.column() == 1:
                value = QVariant(item.getDataType())
            elif index.column() == 2:
                value = QVariant(item.getExtentInPixels().width)
            elif index.column() == 3:
                value = QVariant(item.getExtentInPixels().height)
            elif index.column() == 4:
                value = QVariant(f'{item.getSizeInBytes() / (1024 * 1024):.2f}')

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
