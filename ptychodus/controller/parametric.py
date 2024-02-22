from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import Decimal
from typing import Final
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout, QSpinBox,
                             QWidget)

from ..api.geometry import Interval
from ..api.observer import Observable, Observer
from ..api.parametric import IntegerParameter, RealParameter
from ..view.widgets import DecimalLineEdit, DecimalSlider, LengthWidget

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
        self._widget = QSpinBox()

        self._syncModelToView()
        self._widget.valueChanged.connect(parameter.setValue)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None:
            logger.error('Minimum not provided!')
        else:
            self._widget.blockSignals(True)

            if maximum is None:
                self._widget.setRange(minimum, SpinBoxParameterViewController.MAX_INT)
            else:
                self._widget.setRange(minimum, maximum)

            self._widget.setValue(self._parameter.getValue())
            self._widget.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class DecimalLineEditParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: RealParameter) -> None:
        self._parameter = parameter
        self._widget = DecimalLineEdit.createInstance()

        self._syncModelToView()
        self._widget.valueChanged.connect(parameter.setValue)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, value: Decimal) -> None:
        self._parameter.setValue(float(value))

    def _syncModelToView(self) -> None:
        self._widget.setValue(Decimal(repr(self._parameter.getValue())))

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class DecimalSliderParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: RealParameter) -> None:
        self._parameter = parameter
        self._widget = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        self._syncModelToView()
        self._widget.valueChanged.connect(parameter.setValue)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None or maximum is None:
            logger.error('Range not provided!')
        else:
            value = Decimal(repr(self._parameter.getValue()))
            range_ = Interval[Decimal](Decimal(repr(minimum)), Decimal(repr(maximum)))
            self._widget.setValueAndRange(value, range_)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class LengthWidgetParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: RealParameter) -> None:
        self._parameter = parameter
        self._widget = LengthWidget.createInstance()

        self._syncModelToView()
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


class ParameterDialog(QDialog):

    def __init__(self, viewControllers: Sequence[ParameterViewController],
                 buttonBox: QDialogButtonBox, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._viewControllers = viewControllers
        self._buttonBox = buttonBox

        buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        buttonBox.clicked.connect(self._handleButtonBoxClicked)

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        # FIXME observer -> signal adapter
        # TODO remove observers from viewControllers

        if self._buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class ParameterDialogBuilder:

    def __init__(self) -> None:
        self._viewControllers: dict[str, ParameterViewController] = dict()

    def addSpinBox(self, label: str, parameter: IntegerParameter) -> None:
        self._viewControllers[label] = SpinBoxParameterViewController(parameter)

    def addDecimalLineEdit(self, label: str, parameter: RealParameter) -> None:
        self._viewControllers[label] = DecimalLineEditParameterViewController(parameter)

    def addDecimalSlider(self, label: str, parameter: RealParameter) -> None:
        self._viewControllers[label] = DecimalSliderParameterViewController(parameter)

    def addLengthWidget(self, label: str, parameter: RealParameter) -> None:
        self._viewControllers[label] = LengthWidgetParameterViewController(parameter)

    def build(self, windowTitle: str, parent: QWidget | None) -> QDialog:
        layout = QFormLayout()

        for label, vc in self._viewControllers.items():
            layout.addRow(f'{label}:', vc.getWidget())

        buttonBox = QDialogButtonBox()
        layout.addRow(buttonBox)

        dialog = ParameterDialog(list(self._viewControllers.values()), buttonBox, parent)
        dialog.setLayout(layout)
        dialog.setWindowTitle(windowTitle)

        self._viewControllers.clear()

        return dialog
