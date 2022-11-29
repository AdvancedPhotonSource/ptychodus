from typing import Any, Optional

import numpy
import numpy.typing

from PyQt5.QtCore import Qt, QAbstractTableModel, QDir, QModelIndex, QObject, QVariant


class DataArrayTableModel(QAbstractTableModel):

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._array: Optional[numpy.typing.NDArray[Any]] = None

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> Any:
        result = None

        if role == Qt.DisplayRole:
            result = section

        return QVariant(result)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        result = None

        if index.isValid() and role == Qt.DisplayRole and self._array is not None:
            result = str(self._array[index.row(), index.column()])

        return QVariant(result)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        count = 0

        if self._array is not None:
            count = self._array.shape[0]

        return count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        count = 0

        if self._array is not None:
            count = self._array.shape[1]

        return count

    def setArray(self, data: Optional[numpy.typing.NDArray[Any]]) -> None:
        self.beginResetModel()
        self._array = None

        if data is not None:
            array = numpy.atleast_2d(data)
            self._array = array.T if numpy.ndim(data) == 1 else array

        self.endResetModel()
