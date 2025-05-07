from __future__ import annotations
from decimal import Decimal

import numpy

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

from ptychodus.api.geometry import Interval


class DecimalSlider(QWidget):
    value_changed = pyqtSignal(Decimal)

    def __init__(self, slider: QSlider, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._slider = slider
        self._label = QLabel()
        self._value = Decimal()
        self._minimum = Decimal()
        self._maximum = Decimal()

    @classmethod
    def create_instance(
        cls,
        orientation: Qt.Orientation,
        parent: QWidget | None = None,
        *,
        num_ticks: int = 1000,
    ) -> DecimalSlider:
        slider = QSlider(orientation)
        slider.setRange(0, num_ticks)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(100)

        widget = cls(slider, parent)
        slider.valueChanged.connect(lambda value: widget._set_value_from_slider())
        widget.set_value_and_range(Decimal(1) / 2, Interval[Decimal](Decimal(0), Decimal(1)))

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget._slider)
        layout.addWidget(widget._label)
        widget.setLayout(layout)

        return widget

    def get_value(self) -> Decimal:
        return self._value

    def set_value(self, value: Decimal) -> None:
        if self._set_value_to_slider(value):
            self._emit_value_changed()

    def set_value_and_range(
        self,
        value: Decimal,
        range_: Interval[Decimal],
        block_value_changed_signal: bool = False,
    ) -> None:
        should_emit = False

        if range_.upper <= range_.lower:
            raise ValueError(f'maximum <= minimum ({range_.upper} <= {range_.lower})')

        if range_.lower != self._minimum:
            self._minimum = range_.lower
            should_emit = True

        if range_.upper != self._maximum:
            self._maximum = range_.upper
            should_emit = True

        if self._set_value_to_slider(value):
            should_emit = True

        if not block_value_changed_signal and should_emit:
            self._emit_value_changed()

    def _set_value_from_slider(self) -> None:
        upper = Decimal(self._slider.value() - self._slider.minimum())
        lower = Decimal(self._slider.maximum() - self._slider.minimum())
        alpha = upper / lower
        value = (1 - alpha) * self._minimum + alpha * self._maximum

        if value != self._value:
            self._value = value
            self._update_label()
            self._emit_value_changed()

    def _set_value_to_slider(self, value: Decimal) -> bool:
        should_emit = False

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
            self._update_label()
            should_emit = True

        return should_emit

    def _update_label(self) -> None:
        self._label.setText(f'{self._value:.3f}')

    def _emit_value_changed(self) -> None:
        self.value_changed.emit(self._value)
