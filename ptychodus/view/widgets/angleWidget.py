from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QWidget
import numpy

from .decimalLineEdit import DecimalLineEdit


class AngleWidget(QWidget):
    angleChanged = pyqtSignal(Decimal)

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.angleInTurns = Decimal()
        self.angleLineEdit = DecimalLineEdit.createNonNegativeInstance()
        self.unitsComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> AngleWidget:
        widget = cls(parent)

        widget.angleLineEdit.valueChanged.connect(widget._setAngleInTurnsFromWidgets)

        widget.unitsComboBox.addItem('turn', Decimal(1))
        widget.unitsComboBox.addItem('deg', Decimal(360))
        widget.unitsComboBox.addItem('rad', 2 * Decimal(repr(numpy.pi)))
        widget.unitsComboBox.activated.connect(widget._updateDisplay)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget.angleLineEdit)
        layout.addWidget(widget.unitsComboBox)
        widget.setLayout(layout)

        return widget

    def isReadOnly(self) -> bool:
        return self.angleLineEdit.isReadOnly()

    def setReadOnly(self, enable: bool) -> None:
        self.angleLineEdit.setReadOnly(enable)
        self.unitsComboBox.setEnabled(not enable)

    def getAngleInTurns(self) -> Decimal:
        return self.angleInTurns

    def setAngleInTurns(self, angleInTurns: Decimal) -> None:
        self.angleInTurns = angleInTurns
        self._updateDisplay()
        self.angleChanged.emit(self.getAngleInTurns())

    def _setAngleInTurnsFromWidgets(self, angle: Decimal) -> None:
        self.angleInTurns = angle / self.unitsComboBox.currentData()
        self.angleChanged.emit(self.angleInTurns)

    def _updateDisplay(self) -> None:
        angleInDisplayUnits = self.angleInTurns * self.unitsComboBox.currentData()
        self.angleLineEdit.setValue(angleInDisplayUnits)
