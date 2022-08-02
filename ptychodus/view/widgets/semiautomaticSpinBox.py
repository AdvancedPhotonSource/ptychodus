from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QSizePolicy, QSpinBox, QWidget


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
