from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Final
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout, QSpinBox,
                             QWidget)

from ..api.geometry import Interval
from ..api.observer import Observable, Observer
from ..api.parametric import IntegerParameter, RealParameter
from ..view.widgets import DecimalSlider, LengthWidget

logger = logging.getLogger(__name__)

__all__ = [
    'ParameterDialogBuilder',
]


class ParameterViewController(ABC):

    @abstractmethod
    def getWidget(self) -> QWidget:
        pass


class SpinBoxParameterViewController(ParameterViewController, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, parameter: IntegerParameter) -> None:
        self._parameter = parameter
        self._spinBox = QSpinBox()
        self._spinBox.valueChanged.connect(parameter.setValue)

    def getWidget(self) -> QWidget:
        return self._spinBox

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None:
            logger.error('Minimum not provided!')
        else:
            self._spinBox.blockSignals(True)

            if maximum is None:
                self._spinBox.setRange(minimum, SpinBoxParameterViewController.MAX_INT)
            else:
                self._spinBox.setRange(minimum, maximum)

            self._spinBox.setValue(self._parameter.getValue())
            self._spinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class DecimalSliderParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: RealParameter) -> None:
        self._parameter = parameter
        self._slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self._slider.valueChanged.connect(parameter.setValue)

    def getWidget(self) -> QWidget:
        return self._slider

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None or maximum is None:
            logger.error('Range not provided!')
        else:
            value = Decimal(repr(self._parameter.getValue()))
            range_ = Interval[Decimal](Decimal(repr(minimum)), Decimal(repr(maximum)))
            self._slider.setValueAndRange(value, range_)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class LengthWidgetParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: RealParameter) -> None:
        self._parameter = parameter
        self._widget = LengthWidget.createInstance()
        self._widget.lengthChanged.connect(self._syncViewToModel)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, value: Decimal) -> None:
        self._parameter.setValue(float(value))

    def _syncModelToView(self) -> None:
        self._widget.setLengthInMeters(Decimal(repr(self._parameter.getValue())))

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class ParameterDialogBuilder:

    def __init__(self) -> None:
        self._vcDict: dict[str, ParameterViewController] = dict()
        self._dialog = QDialog()
        self._buttonBox = QDialogButtonBox()
        self._buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        self._buttonBox.clicked.connect(self._handleButtonBoxClicked)

    def addSpinBox(self, label: str, parameter: IntegerParameter) -> None:
        self._vcDict[label] = SpinBoxParameterViewController(parameter)

    def addDecimalSlider(self, label: str, parameter: RealParameter) -> None:
        self._vcDict[label] = DecimalSliderParameterViewController(parameter)

    def addLengthWidget(self, label: str, parameter: RealParameter) -> None:
        self._vcDict[label] = LengthWidgetParameterViewController(parameter)

    def build(self, windowTitle: str, parent: QWidget | None) -> QDialog:
        layout = QFormLayout()

        for label, vc in self._vcDict.items():
            layout.addRow(f'{label}:', vc.getWidget())

        layout.addRow(self._buttonBox)

        if parent is not None:
            self._dialog.setParent(parent)

        self._dialog.setLayout(layout)
        self._dialog.setWindowTitle(windowTitle)
        return self._dialog

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        # FIXME add/remove observers

        if self._buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self._dialog.accept()
        else:
            self._dialog.reject()
