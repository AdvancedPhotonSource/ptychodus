from __future__ import annotations
from decimal import Decimal
import logging

import numpy

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLineEdit, QWidget

logger = logging.getLogger(__name__)


class BottomTitledGroupBox(QGroupBox):
    def __init__(self, title: str, parent: QWidget = None) -> None:
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: bottom center;
            }""")


class LengthWidget(QWidget):
    lengthChanged = pyqtSignal(Decimal)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.lengthInMeters = Decimal()
        self.magnitudeValidator = QDoubleValidator()
        self.magnitudeLineEdit = QLineEdit()
        self.unitsComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> LengthWidget:
        widget = cls(parent)

        widget.magnitudeValidator.setBottom(0.)
        widget.magnitudeLineEdit.setValidator(widget.magnitudeValidator)
        widget.magnitudeLineEdit.editingFinished.connect(widget._setLengthInMetersFromWidgets)

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

    def getLengthInMeters(self) -> Decimal:
        return self.lengthInMeters

    def setLengthInMeters(self, lengthInMeters: Decimal) -> None:
        self.lengthInMeters = lengthInMeters
        exponent = 3 * numpy.floor(numpy.log10(lengthInMeters) / 3)
        index = self.unitsComboBox.findData(exponent)

        if index != -1:
            self.unitsComboBox.setCurrentIndex(index)

        self._updateDisplay()
        self.lengthChanged.emit(self.getLengthInMeters())

    @property
    def _scaleToMeters(self) -> Decimal:
        exponent = self.unitsComboBox.currentData()
        return Decimal(f'1e{exponent:+d}')

    def _setLengthInMetersFromWidgets(self) -> None:
        decimalText = self.magnitudeLineEdit.text()

        try:
            magnitude = Decimal(decimalText)
        except ValueError:
            logger.error(f'Failed to parse length magnitude "{decimalText}"')
        else:
            self.lengthInMeters = magnitude * self._scaleToMeters
            self.lengthChanged.emit(self.lengthInMeters)

    def _updateDisplay(self) -> None:
        lengthInDisplayUnits = self.lengthInMeters / self._scaleToMeters
        self.magnitudeLineEdit.setText(str(lengthInDisplayUnits))


class EnergyWidget(QWidget):
    energyChanged = pyqtSignal(Decimal)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.energyInElectronVolts = Decimal()
        self.magnitudeValidator = QDoubleValidator()
        self.magnitudeLineEdit = QLineEdit()
        self.unitsComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> EnergyWidget:
        widget = cls(parent)

        widget.magnitudeValidator.setBottom(0.)
        widget.magnitudeLineEdit.setValidator(widget.magnitudeValidator)
        widget.magnitudeLineEdit.editingFinished.connect(
            widget._setEnergyInElectronVoltsFromWidgets)

        widget.unitsComboBox.addItem('eV', 0)
        widget.unitsComboBox.addItem('keV', 3)
        widget.unitsComboBox.setCurrentIndex(widget.unitsComboBox.findText('keV'))
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
        exponent = 3 * numpy.floor(numpy.log10(energyInElectronVolts) / 3)
        index = self.unitsComboBox.findData(exponent)

        if index != -1:
            self.unitsComboBox.setCurrentIndex(index)

        self._updateDisplay()
        self.energyChanged.emit(self.getEnergyInElectronVolts())

    @property
    def _scaleToElectronVolts(self) -> Decimal:
        exponent = self.unitsComboBox.currentData()
        return Decimal(f'1e{exponent:+d}')

    def _setEnergyInElectronVoltsFromWidgets(self) -> None:
        decimalText = self.magnitudeLineEdit.text()

        try:
            magnitude = Decimal(decimalText)
        except ValueError:
            logger.error(f'Failed to parse energy magnitude "{decimalText}"')
        else:
            self.energyInElectronVolts = magnitude * self._scaleToElectronVolts
            self.energyChanged.emit(self.energyInElectronVolts)

    def _updateDisplay(self) -> None:
        energyInDisplayUnits = self.energyInElectronVolts / self._scaleToElectronVolts
        self.magnitudeLineEdit.setText(str(energyInDisplayUnits))
