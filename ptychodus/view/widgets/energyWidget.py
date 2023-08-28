from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QWidget

from .decimalLineEdit import DecimalLineEdit


class EnergyWidget(QWidget):
    energyChanged = pyqtSignal(Decimal)

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.energyInElectronVolts = Decimal()
        self.magnitudeLineEdit = DecimalLineEdit.createInstance(isSigned=False)
        self.unitsComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> EnergyWidget:
        widget = cls(parent)

        widget.magnitudeLineEdit.setMinimum(Decimal())
        widget.magnitudeLineEdit.valueChanged.connect(widget._setEnergyInElectronVoltsFromWidgets)

        widget.unitsComboBox.addItem('eV', 0)
        widget.unitsComboBox.addItem('keV', 3)
        widget.unitsComboBox.activated.connect(widget._updateDisplay)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget.magnitudeLineEdit)
        layout.addWidget(widget.unitsComboBox)
        widget.setLayout(layout)

        return widget

    def getEnergyInElectronVolts(self) -> Decimal:
        return self.energyInElectronVolts

    def setEnergyInElectronVolts(self, energyInElectronVolts: Decimal) -> None:
        self.energyInElectronVolts = energyInElectronVolts

        if energyInElectronVolts > Decimal():
            exponent = 3 * int(
                (energyInElectronVolts.log10() / 3).to_integral_exact(rounding=ROUND_FLOOR))
            index = self.unitsComboBox.findData(exponent)

            if index != -1:
                self.unitsComboBox.setCurrentIndex(index)

        self._updateDisplay()
        self.energyChanged.emit(self.getEnergyInElectronVolts())

    @property
    def _scaleToElectronVolts(self) -> Decimal:
        exponent = self.unitsComboBox.currentData()
        return Decimal(f'1e{exponent:+d}')

    def _setEnergyInElectronVoltsFromWidgets(self, magnitude: Decimal) -> None:
        self.energyInElectronVolts = magnitude * self._scaleToElectronVolts
        self.energyChanged.emit(self.energyInElectronVolts)

    def _updateDisplay(self) -> None:
        energyInDisplayUnits = self.energyInElectronVolts / self._scaleToElectronVolts
        self.magnitudeLineEdit.setValue(energyInDisplayUnits)
