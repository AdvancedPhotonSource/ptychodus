from __future__ import annotations
from decimal import Decimal

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QWidget
import numpy

from .decimal_line_edit import DecimalLineEdit


class AngleWidget(QWidget):
    angle_changed = pyqtSignal(Decimal)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.angle_in_turns = Decimal()
        self.angle_line_edit = DecimalLineEdit.create_instance(is_signed=False)
        self.units_combo_box = QComboBox()

        self.angle_line_edit.value_changed.connect(self._set_angle_in_turns_from_widgets)

        self.units_combo_box.addItem('turn', Decimal(1))
        self.units_combo_box.addItem('deg', Decimal(360))
        self.units_combo_box.addItem('rad', 2 * Decimal.from_float(numpy.pi))
        self.units_combo_box.activated.connect(self._update_display)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.angle_line_edit)
        layout.addWidget(self.units_combo_box)
        self.setLayout(layout)

    def is_read_only(self) -> bool:
        return self.angle_line_edit.is_read_only()

    def set_read_only(self, enable: bool) -> None:
        self.angle_line_edit.set_read_only(enable)

    def get_angle_in_turns(self) -> Decimal:
        return self.angle_in_turns

    def set_angle_in_turns(self, angle_in_turns: Decimal) -> None:
        self.angle_in_turns = angle_in_turns
        self._update_display()
        self.angle_changed.emit(self.get_angle_in_turns())

    def _set_angle_in_turns_from_widgets(self, angle: Decimal) -> None:
        self.angle_in_turns = angle / self.units_combo_box.currentData()
        self.angle_changed.emit(self.angle_in_turns)

    def _update_display(self) -> None:
        angle_in_display_units = self.angle_in_turns * self.units_combo_box.currentData()
        self.angle_line_edit.set_value(angle_in_display_units)
