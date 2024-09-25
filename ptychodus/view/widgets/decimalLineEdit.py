from __future__ import annotations
from decimal import Decimal
import logging

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QHBoxLayout, QLineEdit, QWidget

logger = logging.getLogger(__name__)


class DecimalLineEdit(QWidget):
    valueChanged = pyqtSignal(Decimal)

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._validator = QDoubleValidator()
        self._lineEdit = QLineEdit()
        self._value = Decimal()
        self._minimum: Decimal | None = None
        self._maximum: Decimal | None = None

    @classmethod
    def createInstance(cls,
                       *,
                       isSigned: bool = False,
                       parent: QWidget | None = None) -> DecimalLineEdit:
        widget = cls(parent)

        widget._lineEdit.setValidator(widget._validator)
        widget._lineEdit.editingFinished.connect(widget._setValueFromLineEdit)
        widget._setValueToLineEditAndEmitValueChanged()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget._lineEdit)
        widget.setLayout(layout)

        if not isSigned:
            widget._validator.setBottom(0.0)

        return widget

    def isReadOnly(self) -> bool:
        return self._lineEdit.isReadOnly()

    def setReadOnly(self, enable: bool) -> None:
        self._lineEdit.setReadOnly(enable)

    def getValue(self) -> Decimal:
        if self._minimum is not None and self._value < self._minimum:
            return self._minimum

        if self._maximum is not None and self._value > self._maximum:
            return self._maximum

        return self._value

    def setValue(self, value: Decimal) -> None:
        if value != self._value:
            self._value = value
            self._setValueToLineEditAndEmitValueChanged()

    def getMinimum(self) -> Decimal | None:
        return self._minimum

    def setMinimum(self, value: Decimal) -> None:
        valueBefore = self.getValue()
        self._minimum = value
        valueAfter = self.getValue()

        if valueBefore != valueAfter:
            self._setValueToLineEditAndEmitValueChanged()

    def getMaximum(self) -> Decimal | None:
        return self._maximum

    def setMaximum(self, value: Decimal) -> None:
        valueBefore = self.getValue()
        self._maximum = value
        valueAfter = self.getValue()

        if valueBefore != valueAfter:
            self._setValueToLineEditAndEmitValueChanged()

    def _setValueFromLineEdit(self) -> None:
        decimalText = self._lineEdit.text()

        try:
            self._value = Decimal(decimalText)
        except ValueError:
            logger.error(f'Failed to parse value "{decimalText}"')
        else:
            self._emitValueChanged()

    def _setValueToLineEditAndEmitValueChanged(self) -> None:
        self._lineEdit.setText(str(self.getValue()))
        self._emitValueChanged()

    def _emitValueChanged(self) -> None:
        self.valueChanged.emit(self._value)
