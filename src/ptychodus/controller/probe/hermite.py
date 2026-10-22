from __future__ import annotations
from enum import IntEnum
from typing import Any
import logging

import numpy

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject

from ...model.product.probe import HermiteProbeBuilder

logger = logging.getLogger(__name__)


class _Column(IntEnum):
    """Column order of HermiteTableModel.

    An enum rather than module-level ints so that ``match`` arms are value patterns:
    a bare name would be a capture pattern that swallows every column.
    """

    ORDER_X = 0
    ORDER_Y = 1
    AMPLITUDE = 2
    PHASE_TR = 3


class HermiteTableModel(QAbstractTableModel):
    def __init__(self, builder: HermiteProbeBuilder, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._builder = builder
        self._header = [
            'Order X',
            'Order Y',
            'Amplitude',
            'Phase [tr]',
        ]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() in (_Column.AMPLITUDE, _Column.PHASE_TR):
            value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        try:
            mode = self._builder.get_mode(index.row())
        except IndexError as err:
            logger.exception(err)
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if index.column() == _Column.ORDER_X:
                return mode.order_x
            elif index.column() == _Column.ORDER_Y:
                return mode.order_y
            elif index.column() == _Column.AMPLITUDE:
                return f'{numpy.absolute(mode.coefficient):.6g}'
            elif index.column() == _Column.PHASE_TR:
                return f'{numpy.angle(mode.coefficient):.6g}'

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
            return False

        if role == Qt.ItemDataRole.EditRole:
            if index.column() == _Column.AMPLITUDE:
                try:
                    amplitude = float(value)
                except ValueError:
                    return False

                try:
                    coef = self._builder.get_mode(index.row()).coefficient
                except IndexError:
                    return False

                try:
                    complex_value = amplitude * coef / numpy.absolute(coef)
                except ZeroDivisionError:
                    complex_value = amplitude + 0j

                self._builder.set_coefficient(index.row(), complex_value)
                return True
            elif index.column() == _Column.PHASE_TR:
                try:
                    phase = float(value)
                except ValueError:
                    return False

                try:
                    coef = self._builder.get_mode(index.row()).coefficient
                except IndexError:
                    return False

                complex_value = numpy.absolute(coef) * numpy.exp(2j * numpy.pi * phase)
                self._builder.set_coefficient(index.row(), complex_value)
                return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._builder)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)
