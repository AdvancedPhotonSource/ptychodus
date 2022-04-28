from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR
from typing import Optional
import logging

import numpy

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QSizePolicy, QSlider, QSpinBox, QWidget

logger = logging.getLogger(__name__)


class BottomTitledGroupBox(QGroupBox):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: bottom center;
            }""")


class DecimalSlider(QWidget):
    valueChanged = pyqtSignal(Decimal)

    def __init__(self, orientation: Qt.Orientation, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self._slider = QSlider(orientation)
        self._label = QLabel()
        self._value = Decimal()
        self._minimum = Decimal()
        self._maximum = Decimal()

    @classmethod
    def createInstance(cls,
                       orientation: Qt.Orientation,
                       parent: Optional[QWidget] = None) -> DecimalSlider:
        widget = cls(orientation, parent)

        widget._slider.setRange(0, 1000)
        widget._slider.setTickPosition(QSlider.TicksBelow)
        widget._slider.setTickInterval(100)
        widget._slider.valueChanged.connect(lambda value: widget._setValueFromSlider())
        widget.setValueAndRange(Decimal(1) / 2, Decimal(0), Decimal(1))

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget._slider)
        layout.addWidget(widget._label)
        widget.setLayout(layout)

        return widget

    def getValue(self) -> Decimal:
        return self._value

    def setValue(self, value: Decimal) -> None:
        if self._setValueToSlider(value):
            self._emitValueChanged()

    def setValueAndRange(self,
                         value: Decimal,
                         minimum: Decimal,
                         maximum: Decimal,
                         blockValueChangedSignal: bool = False) -> None:
        shouldEmit = False

        if maximum <= minimum:
            raise ValueError('maximum <= minimum')

        if minimum != self._minimum:
            self._minimum = Decimal(minimum)
            shouldEmit = True

        if maximum != self._maximum:
            self._maximum = Decimal(maximum)
            shouldEmit = True

        if self._setValueToSlider(value):
            shouldEmit = True

        if not blockValueChangedSignal and shouldEmit:
            self._emitValueChanged()

    def _setValueFromSlider(self) -> None:
        upper = Decimal(self._slider.value() - self._slider.minimum())
        lower = Decimal(self._slider.maximum() - self._slider.minimum())
        alpha = upper / lower
        value = (1 - alpha) * self._minimum + alpha * self._maximum

        if value != self._value:
            self._value = value
            self._updateLabel()
            self._emitValueChanged()

    def _setValueToSlider(self, value: Decimal) -> bool:
        shouldEmit = False

        alpha = (Decimal(value) - self._minimum) / (self._maximum - self._minimum)
        ivaluef = (1 - alpha) * self._slider.minimum() + alpha * self._slider.maximum()
        ivalue = int(numpy.rint(float(ivaluef)))

        if value < self._minimum:
            value = self._minimum
            ivalue = self._slider.minimum()
        elif value > self._maximum:
            value = self._maximum
            ivalue = self._slider.maximum()

        self._slider.blockSignals(True)
        self._slider.setValue(ivalue)
        self._slider.blockSignals(False)

        if value != self._value:
            self._value = Decimal(value)
            self._updateLabel()
            shouldEmit = True

        return shouldEmit

    def _updateLabel(self) -> None:
        self._label.setText(f'{self._value:.3f}')

    def _emitValueChanged(self) -> None:
        self.valueChanged.emit(self._value)


class SemiautomaticSpinBox(QWidget):
    valueChanged = pyqtSignal(int)
    autoToggled = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.spinBox = QSpinBox()
        self.autoCheckBox = QCheckBox('Auto')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SemiautomaticSpinBox:
        widget = cls(parent)

        widget.spinBox.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        widget.spinBox.valueChanged.connect(widget.valueChanged.emit)
        widget.autoCheckBox.toggled.connect(widget._handleAutoToggled)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget.spinBox)
        layout.addWidget(widget.autoCheckBox)
        widget.setLayout(layout)

        return widget

    def _handleAutoToggled(self, checked: bool) -> None:
        self.spinBox.setEnabled(not checked)
        self.autoToggled.emit(checked)

    def getValue(self) -> int:
        return self.spinBox.value()

    def setValue(self, val: int) -> None:
        self.spinBox.setValue(val)

    def setValueAndRange(self, value: int, minimum: int, maximum: int) -> None:
        shouldEmit = (self.spinBox.value() == value)

        self.spinBox.blockSignals(True)
        self.spinBox.setRange(minimum, maximum)
        self.spinBox.setValue(value)
        self.spinBox.blockSignals(False)

        if shouldEmit:
            self.valueChanged.emit(value)

    def isAutomatic(self) -> bool:
        return self.autoCheckBox.isChecked()

    def setAutomatic(self, isAuto: bool) -> None:
        self.autoCheckBox.setChecked(isAuto)


class LengthWidget(QWidget):
    lengthChanged = pyqtSignal(Decimal)

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.lengthInMeters = Decimal()
        self.magnitudeValidator = QDoubleValidator()
        self.magnitudeLineEdit = QLineEdit()
        self.unitsComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> LengthWidget:
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

        if lengthInMeters > 0:
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

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.energyInElectronVolts = Decimal()
        self.magnitudeValidator = QDoubleValidator()
        self.magnitudeLineEdit = QLineEdit()
        self.unitsComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> EnergyWidget:
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

        if energyInElectronVolts > 0:
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
