from __future__ import annotations
from enum import Enum
from typing import Any
import logging

import numpy

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject

from ...model.product.probe import HermiteProbeBuilder

logger = logging.getLogger(__name__)


class _Column(Enum):
    """Every column of HermiteTableModel: its order, its header and whether it edits.

    The member value pairs the header with editability, and the position comes
    from declaration order rather than being written out, so a column cannot be added
    to the ordering and forgotten in the labels or the editable set. Nothing could pin
    that alignment: CLAUDE.md puts this file outside the tested surface.

    Bare names in a ``match`` arm are capture patterns, not value patterns.
    """

    ORDER_X = ('Order X', False)
    ORDER_Y = ('Order Y', False)
    AMPLITUDE = ('Amplitude', True)
    PHASE_TR = ('Phase [tr]', True)

    def __init__(self, header: str, editable: bool) -> None:
        self.header = header
        self.editable = editable
        # Members already defined when this one is constructed, which is its position.
        self.index = len(self.__class__.__members__)

    @classmethod
    def from_index(cls, index: int) -> _Column:
        """The column at `index`.

        Raises:
            IndexError: for a column outside the table, which the model's own
                ``columnCount`` makes impossible and so would be a bug rather than input.
        """
        if index < 0:
            raise IndexError(f'Column index {index} is negative.')

        return list(cls)[index]


class HermiteTableModel(QAbstractTableModel):
    def __init__(self, builder: HermiteProbeBuilder, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._builder = builder

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and _Column.from_index(index.column()).editable:
            value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return _Column.from_index(section).header

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        try:
            mode = self._builder.get_mode(index.row())
        except IndexError as err:
            logger.exception(err)
            return None

        column = _Column.from_index(index.column())

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if column is _Column.ORDER_X:
                return mode.order_x
            elif column is _Column.ORDER_Y:
                return mode.order_y
            elif column is _Column.AMPLITUDE:
                return f'{numpy.absolute(mode.coefficient):.6g}'
            elif column is _Column.PHASE_TR:
                return f'{numpy.angle(mode.coefficient):.6g}'

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid():
            return False

        column = _Column.from_index(index.column())

        if role == Qt.ItemDataRole.EditRole:
            if column is _Column.AMPLITUDE:
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
            elif column is _Column.PHASE_TR:
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
        return len(_Column)
