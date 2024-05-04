from __future__ import annotations
from decimal import Decimal

import numpy

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

from ...api.geometry import Interval


class DecimalSlider(QWidget):
    valueChanged = pyqtSignal(Decimal)

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._slider = QSlider(orientation)
        self._label = QLabel()
        self._value = Decimal()
        self._minimum = Decimal()
        self._maximum = Decimal()

    @classmethod
    def createInstance(cls,
                       orientation: Qt.Orientation,
                       parent: QWidget | None = None) -> DecimalSlider:
        widget = cls(orientation, parent)

        widget._slider.setRange(0, 1000)
        widget._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        widget._slider.setTickInterval(100)
        widget._slider.valueChanged.connect(lambda value: widget._setValueFromSlider())
        widget.setValueAndRange(Decimal(1) / 2, Interval[Decimal](Decimal(0), Decimal(1)))

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
                         range_: Interval[Decimal],
                         blockValueChangedSignal: bool = False) -> None:
        shouldEmit = False

        if range_.upper <= range_.lower:
            raise ValueError(f'maximum <= minimum ({range_.upper} <= {range_.lower})')

        if range_.lower != self._minimum:
            self._minimum = range_.lower
            shouldEmit = True

        if range_.upper != self._maximum:
            self._maximum = range_.upper
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
