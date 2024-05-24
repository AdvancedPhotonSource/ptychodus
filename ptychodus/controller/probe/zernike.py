from typing import Any
import logging

import numpy

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject

from ...model.product.probe import ZernikeProbeBuilder

logger = logging.getLogger(__name__)


class ZernikeTableModel(QAbstractTableModel):

    def __init__(self, builder: ZernikeProbeBuilder, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._builder = builder
        self._header = [
            'Radial Degree',
            'Angular Frequency',
            'Amplitude',
            'Phase [tr]',
        ]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() in (2, 3):
            value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        try:
            poly = self._builder.getPolynomial(index.row())
            coef = self._builder.getCoefficient(index.row())
        except IndexError as err:
            logger.exception(err)
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if index.column() == 0:
                return poly.radial_degree
            elif index.column() == 1:
                return poly.angular_frequency
            elif index.column() == 2:
                return f'{numpy.absolute(coef):.6g}'
            elif index.column() == 3:
                return f'{numpy.angle(coef):.6g}'

    def setData(self,
                index: QModelIndex,
                value: Any,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.EditRole:
            if index.column() == 2:
                try:
                    amplitude = float(value)
                except ValueError:
                    return False

                try:
                    coef = self._builder.getCoefficient(index.row())
                except IndexError:
                    return False

                try:
                    complexValue = amplitude * coef / numpy.absolute(coef)
                except ZeroDivisionError:
                    complexValue = amplitude + 0j

                self._builder.setCoefficient(index.row(), complexValue)
                return True
            elif index.column() == 3:
                try:
                    phase = float(value)
                except ValueError:
                    return False

                try:
                    coef = self._builder.getCoefficient(index.row())
                except IndexError:
                    return False

                complexValue = numpy.absolute(coef) * numpy.exp(2j * numpy.pi * phase)
                self._builder.setCoefficient(index.row(), complexValue)
                return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._builder)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)
