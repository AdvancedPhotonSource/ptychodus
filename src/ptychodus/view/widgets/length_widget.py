from __future__ import annotations
from decimal import Decimal, ROUND_FLOOR

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QWidget

from .decimal_line_edit import DecimalLineEdit


class LengthWidget(QWidget):
    length_changed = pyqtSignal(Decimal)

    def __init__(self, is_signed: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.length_m = Decimal()
        self.line_edit = DecimalLineEdit.create_instance(is_signed=is_signed)
        self.units_combo_box = QComboBox()

        if not is_signed:
            self.line_edit.set_minimum(Decimal())

        self.line_edit.value_changed.connect(self._set_length_m_from_widgets)

        self.units_combo_box.addItem('m', 0)
        self.units_combo_box.addItem('mm', -3)
        self.units_combo_box.addItem('\u00b5m', -6)
        self.units_combo_box.addItem('nm', -9)
        self.units_combo_box.addItem('\u212b', -10)
        self.units_combo_box.addItem('pm', -12)
        self.units_combo_box.activated.connect(self._update_display)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.units_combo_box)
        self.setLayout(layout)

    def is_read_only(self) -> bool:
        return self.line_edit.is_read_only()

    def set_read_only(self, enable: bool) -> None:
        self.line_edit.set_read_only(enable)

    def get_length_m(self) -> Decimal:
        return self.length_m

    def set_length_m(self, length_m: Decimal) -> None:
        self.length_m = length_m

        if not length_m.is_zero():
            exponent = 3 * int((abs(length_m).log10() / 3).to_integral_exact(rounding=ROUND_FLOOR))
            index = self.units_combo_box.findData(exponent)

            if index != -1:
                self.units_combo_box.setCurrentIndex(index)

        self._update_display()

    @property
    def _scale_to_meters(self) -> Decimal:
        exponent = self.units_combo_box.currentData()
        return Decimal(f'1e{exponent:+d}')

    def _set_length_m_from_widgets(self, magnitude: Decimal) -> None:
        self.length_m = magnitude * self._scale_to_meters
        self.length_changed.emit(self.length_m)

    def _update_display(self) -> None:
        length_in_display_units = self.length_m / self._scale_to_meters
        self.line_edit.set_value(length_in_display_units)
