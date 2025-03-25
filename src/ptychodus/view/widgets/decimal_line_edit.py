from __future__ import annotations
from decimal import Decimal
import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QWidget

logger = logging.getLogger(__name__)


class DecimalLineEdit(QWidget):
    value_changed = pyqtSignal(Decimal)

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._validator = QDoubleValidator()
        self._line_edit = QLineEdit()
        self._value = Decimal()
        self._minimum: Decimal | None = None
        self._maximum: Decimal | None = None

    @classmethod
    def create_instance(
        cls, *, is_signed: bool = False, parent: QWidget | None = None
    ) -> DecimalLineEdit:
        widget = cls(parent)

        widget._line_edit.setValidator(widget._validator)
        widget._line_edit.editingFinished.connect(widget._set_value_from_line_edit)
        widget._set_value_to_line_edit_and_emit_value_changed()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget._line_edit)
        widget.setLayout(layout)

        if not is_signed:
            widget._validator.setBottom(0.0)

        return widget

    def is_read_only(self) -> bool:
        return self._line_edit.isReadOnly()

    def set_read_only(self, enable: bool) -> None:
        self._line_edit.setReadOnly(enable)

    def get_value(self) -> Decimal:
        if self._minimum is not None and self._value < self._minimum:
            return self._minimum

        if self._maximum is not None and self._value > self._maximum:
            return self._maximum

        return self._value

    def set_value(self, value: Decimal) -> None:
        if value != self._value:
            self._value = value
            self._set_value_to_line_edit_and_emit_value_changed()

    def get_minimum(self) -> Decimal | None:
        return self._minimum

    def set_minimum(self, value: Decimal) -> None:
        value_before = self.get_value()
        self._minimum = value
        value_after = self.get_value()

        if value_before != value_after:
            self._set_value_to_line_edit_and_emit_value_changed()

    def get_maximum(self) -> Decimal | None:
        return self._maximum

    def set_maximum(self, value: Decimal) -> None:
        value_before = self.get_value()
        self._maximum = value
        value_after = self.get_value()

        if value_before != value_after:
            self._set_value_to_line_edit_and_emit_value_changed()

    def _set_value_from_line_edit(self) -> None:
        decimal_text = self._line_edit.text()

        try:
            self._value = Decimal(decimal_text)
        except ValueError:
            logger.error(f'Failed to parse value "{decimal_text}"')
        else:
            self._emit_value_changed()

    def _set_value_to_line_edit_and_emit_value_changed(self) -> None:
        self._line_edit.setText(str(self.get_value()))
        self._emit_value_changed()

    def _emit_value_changed(self) -> None:
        self.value_changed.emit(self._value)
