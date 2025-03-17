from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QWidget

from .decimalLineEdit import DecimalLineEdit


class LengthWidget(QWidget):
    lengthChanged = pyqtSignal(Decimal)

    def __init__(self, isSigned: bool, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.lengthInMeters = Decimal()
        self.lineEdit = DecimalLineEdit.create_instance(isSigned=isSigned)
        self.unitsComboBox = QComboBox()

    @classmethod
    def create_instance(
        cls, *, is_signed: bool = False, parent: QWidget | None = None
    ) -> LengthWidget:
        widget = cls(is_signed, parent)

        if not is_signed:
            widget.lineEdit.setMinimum(Decimal())

        widget.lineEdit.valueChanged.connect(widget._setLengthInMetersFromWidgets)

        widget.unitsComboBox.addItem('m', 0)
        widget.unitsComboBox.addItem('mm', -3)
        widget.unitsComboBox.addItem('\u00b5m', -6)
        widget.unitsComboBox.addItem('nm', -9)
        widget.unitsComboBox.addItem('\u212b', -10)
        widget.unitsComboBox.addItem('pm', -12)
        widget.unitsComboBox.activated.connect(widget._updateDisplay)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget.lineEdit)
        layout.addWidget(widget.unitsComboBox)
        widget.setLayout(layout)

        return widget

    def isReadOnly(self) -> bool:
        return self.lineEdit.isReadOnly()

    def setReadOnly(self, enable: bool) -> None:
        self.lineEdit.setReadOnly(enable)

    def getLengthInMeters(self) -> Decimal:
        return self.lengthInMeters

    def setLengthInMeters(self, lengthInMeters: Decimal) -> None:
        self.lengthInMeters = lengthInMeters

        if not lengthInMeters.is_zero():
            exponent = 3 * int(
                (abs(lengthInMeters).log10() / 3).to_integral_exact(rounding=ROUND_FLOOR)
            )
            index = self.unitsComboBox.findData(exponent)

            if index != -1:
                self.unitsComboBox.setCurrentIndex(index)

        self._updateDisplay()

    @property
    def _scaleToMeters(self) -> Decimal:
        exponent = self.unitsComboBox.currentData()
        return Decimal(f'1e{exponent:+d}')

    def _setLengthInMetersFromWidgets(self, magnitude: Decimal) -> None:
        self.lengthInMeters = magnitude * self._scaleToMeters
        self.lengthChanged.emit(self.lengthInMeters)

    def _updateDisplay(self) -> None:
        lengthInDisplayUnits = self.lengthInMeters / self._scaleToMeters
        self.lineEdit.setValue(lengthInDisplayUnits)
