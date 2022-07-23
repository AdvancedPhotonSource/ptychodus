from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QWidget

from .decimalLineEdit import DecimalLineEdit


class LengthWidget(QWidget):
    lengthChanged = pyqtSignal(Decimal)

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.lengthInMeters = Decimal()
        self.magnitudeLineEdit = DecimalLineEdit.createNonNegativeInstance()
        self.unitsComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> LengthWidget:
        widget = cls(parent)

        widget.magnitudeLineEdit.setMinimum(Decimal())
        widget.magnitudeLineEdit.valueChanged.connect(widget._setLengthInMetersFromWidgets)

        widget.unitsComboBox.addItem('m', 0)
        widget.unitsComboBox.addItem('mm', -3)
        widget.unitsComboBox.addItem('\u00B5m', -6)
        widget.unitsComboBox.addItem('nm', -9)
        widget.unitsComboBox.activated.connect(widget._updateDisplay)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget.magnitudeLineEdit)
        layout.addWidget(widget.unitsComboBox)
        widget.setLayout(layout)

        return widget

    def isReadOnly(self) -> bool:
        return self.magnitudeLineEdit.isReadOnly()

    def setReadOnly(self, enable: bool) -> None:
        self.magnitudeLineEdit.setReadOnly(enable)
        self.unitsComboBox.setEnabled(not enable)

    def getLengthInMeters(self) -> Decimal:
        return self.lengthInMeters

    def setLengthInMeters(self, lengthInMeters: Decimal) -> None:
        self.lengthInMeters = lengthInMeters

        if lengthInMeters > Decimal():
            exponent = 3 * int(
                (lengthInMeters.log10() / 3).to_integral_exact(rounding=ROUND_FLOOR))
            index = self.unitsComboBox.findData(exponent)

            if index != -1:
                self.unitsComboBox.setCurrentIndex(index)

        self._updateDisplay()
        self.lengthChanged.emit(self.getLengthInMeters())

    @property
    def _scaleToMeters(self) -> Decimal:
        exponent = self.unitsComboBox.currentData()
        return Decimal(f'1e{exponent:+d}')

    def _setLengthInMetersFromWidgets(self, magnitude: Decimal) -> None:
        self.lengthInMeters = magnitude * self._scaleToMeters
        self.lengthChanged.emit(self.lengthInMeters)

    def _updateDisplay(self) -> None:
        lengthInDisplayUnits = self.lengthInMeters / self._scaleToMeters
        self.magnitudeLineEdit.setValue(lengthInDisplayUnits)
