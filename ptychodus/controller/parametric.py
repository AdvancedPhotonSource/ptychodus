from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import Decimal
from typing import Final
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGroupBox, QSpinBox, QWidget)

from ..api.geometry import Interval
from ..api.observer import Observable, Observer
from ..api.parametric import BooleanParameter, IntegerParameter, RealParameter
from ..view.widgets import AngleWidget, DecimalLineEdit, DecimalSlider, LengthWidget

logger = logging.getLogger(__name__)

__all__ = [
    'ParameterDialogBuilder',
]


class ParameterViewController(ABC):

    @abstractmethod
    def getWidget(self) -> QWidget:
        pass


class CheckBoxParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: BooleanParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QCheckBox()

        self._syncModelToView()
        self._widget.toggled.connect(parameter.setValue)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._parameter.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class SpinBoxParameterViewController(ParameterViewController, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, parameter: IntegerParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QSpinBox()

        self._syncModelToView()
        self._widget.valueChanged.connect(parameter.setValue)
        parameter.addObserver(self)

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

    def __init__(self, parameter: RealParameter, *, isSigned: bool = False) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalLineEdit.createInstance(isSigned=isSigned)

        self._syncModelToView()
        self._widget.valueChanged.connect(parameter.setValue)
        parameter.addObserver(self)

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
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        self._syncModelToView()
        self._widget.valueChanged.connect(self._syncViewToModel)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, value: Decimal) -> None:
        self._parameter.setValue(float(value))

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

    def __init__(self, parameter: RealParameter, *, isSigned: bool = False) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = LengthWidget.createInstance(isSigned=isSigned)

        self._syncModelToView()
        self._widget.lengthChanged.connect(self._syncViewToModel)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, value: Decimal) -> None:
        self._parameter.setValue(float(value))

    def _syncModelToView(self) -> None:
        self._widget.setLengthInMeters(Decimal(repr(self._parameter.getValue())))

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class AngleWidgetParameterViewController(ParameterViewController, Observer):

    def __init__(self, parameter: RealParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = AngleWidget.createInstance()

        self._syncModelToView()
        self._widget.angleChanged.connect(self._syncViewToModel)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, value: Decimal) -> None:
        self._parameter.setValue(float(value))

    def _syncModelToView(self) -> None:
        self._widget.setAngleInTurns(Decimal(repr(self._parameter.getValue())))

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
        # TODO remove observers from viewControllers

        if self._buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class ParameterDialogBuilder:

    def __init__(self) -> None:
        self._viewControllers: dict[tuple[str, str], ParameterViewController] = dict()

    def addCheckBox(self, parameter: BooleanParameter, label: str, group: str = '') -> None:
        viewController = CheckBoxParameterViewController(parameter)
        self.addViewController(viewController, label, group)

    def addSpinBox(self, parameter: IntegerParameter, label: str, group: str = '') -> None:
        viewController = SpinBoxParameterViewController(parameter)
        self.addViewController(viewController, label, group)

    def addDecimalLineEdit(self, parameter: RealParameter, label: str, group: str = '') -> None:
        viewController = DecimalLineEditParameterViewController(parameter)
        self.addViewController(viewController, label, group)

    def addDecimalSlider(self, parameter: RealParameter, label: str, group: str = '') -> None:
        viewController = DecimalSliderParameterViewController(parameter)
        self.addViewController(viewController, label, group)

    def addLengthWidget(self, parameter: RealParameter, label: str, group: str = '') -> None:
        viewController = LengthWidgetParameterViewController(parameter)
        self.addViewController(viewController, label, group)

    def addAngleWidget(self, parameter: RealParameter, label: str, group: str = '') -> None:
        viewController = AngleWidgetParameterViewController(parameter)
        self.addViewController(viewController, label, group)

    def addViewController(self,
                          viewController: ParameterViewController,
                          label: str,
                          group: str = '') -> None:
        self._viewControllers[group, label] = viewController

    def build(self, windowTitle: str, parent: QWidget | None) -> QDialog:
        groupDict = {'': QFormLayout()}

        for (groupName, widgetLabel), vc in self._viewControllers.items():
            try:
                layout = groupDict[groupName]
            except KeyError:
                layout = QFormLayout()
                groupDict[groupName] = layout

            widget = vc.getWidget()

            if isinstance(widget, CheckBoxParameterViewController):
                widget.setText(widgetLabel)
                layout.addRow(widget)
            elif widgetLabel.startswith('_'):
                layout.addRow(widget)
            else:
                layout.addRow(widgetLabel, widget)

        buttonBox = QDialogButtonBox()
        layout = groupDict.pop('')

        for groupName, groupLayout in groupDict.items():
            groupBox = QGroupBox(groupName)
            groupBox.setLayout(groupLayout)
            layout.addRow(groupBox)

        layout.addRow(buttonBox)

        dialog = ParameterDialog(list(self._viewControllers.values()), buttonBox, parent)
        dialog.setLayout(layout)
        dialog.setWindowTitle(windowTitle)

        self._viewControllers.clear()

        return dialog
