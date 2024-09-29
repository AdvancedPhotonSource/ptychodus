from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import Decimal
from typing import Final
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QValidator
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ..view.widgets import AngleWidget, DecimalLineEdit, DecimalSlider, LengthWidget

logger = logging.getLogger(__name__)

__all__ = [
    'ParameterViewBuilder',
]


class ParameterViewController(ABC):
    @abstractmethod
    def getWidget(self) -> QWidget:
        pass


class CheckBoxParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: BooleanParameter, text: str, *, tooltip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QCheckBox(text)

        if tooltip:
            self._widget.setToolTip(tooltip)

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

    def __init__(self, parameter: IntegerParameter, *, tooltip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QSpinBox()

        if tooltip:
            self._widget.setToolTip(tooltip)

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


class ComboBoxParameterViewController(ParameterViewController, Observer):
    def __init__(
        self, parameter: StringParameter, items: Sequence[str], *, tooltip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QComboBox()

        for item in items:
            self._widget.addItem(item)

        if tooltip:
            self._widget.setToolTip(tooltip)

        self._syncModelToView()
        self._widget.textActivated.connect(parameter.setValue)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setCurrentText(self._parameter.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class LineEditParameterViewController(ParameterViewController, Observer):
    def __init__(
        self, parameter: StringParameter, validator: QValidator | None = None, *, tooltip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QLineEdit()

        if validator is not None:
            self._widget.setValidator(validator)

        if tooltip:
            self._widget.setToolTip(tooltip)

        self._syncModelToView()
        self._widget.editingFinished.connect(self._syncViewToModel)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self) -> None:
        self._parameter.setValue(self._widget.text())

    def _syncModelToView(self) -> None:
        self._widget.setText(self._parameter.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class IntegerLineEditParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: IntegerParameter, *, tooltip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QLineEdit()

        if tooltip:
            self._widget.setToolTip(tooltip)

        validator = QIntValidator()
        bottom = parameter.getMinimum()
        top = parameter.getMaximum()

        if bottom is not None:
            validator.setBottom(bottom)

        if top is not None:
            validator.setTop(top)

        self._widget.setValidator(validator)

        self._syncModelToView()
        self._widget.editingFinished.connect(self._syncViewToModel)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self) -> None:
        text = self._widget.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert "{text}" to int!')
        else:
            self._parameter.setValue(value)

    def _syncModelToView(self) -> None:
        self._widget.setText(str(self._parameter.getValue()))

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class DecimalLineEditParameterViewController(ParameterViewController, Observer):
    def __init__(
        self, parameter: RealParameter, *, isSigned: bool = False, tooltip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalLineEdit.createInstance(isSigned=isSigned)

        if tooltip:
            self._widget.setToolTip(tooltip)

        self._syncModelToView()
        self._widget.valueChanged.connect(self._syncViewToModel)
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
    def __init__(self, parameter: RealParameter, *, tooltip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        if tooltip:
            self._widget.setToolTip(tooltip)

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


class ParameterWidget(QWidget):
    def __init__(
        self, viewControllers: Sequence[ParameterViewController], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._viewControllers = viewControllers


class ParameterDialog(QDialog):
    def __init__(
        self,
        viewControllers: Sequence[ParameterViewController],
        buttonBox: QDialogButtonBox,
        parent: QWidget | None,
    ) -> None:
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


class ParameterViewBuilder:
    def __init__(self) -> None:
        self._viewControllersTop: list[ParameterViewController] = list()
        self._viewControllers: dict[tuple[str, str], ParameterViewController] = dict()
        self._viewControllersBottom: list[ParameterViewController] = list()

    def addCheckBox(
        self,
        parameter: BooleanParameter,
        label: str,
        *,
        tooltip: str = '',
        group: str = '',
    ) -> None:
        viewController = CheckBoxParameterViewController(parameter, '')
        self.addViewController(viewController, label, tooltip=tooltip, group=group)

    def addSpinBox(
        self,
        parameter: IntegerParameter,
        label: str,
        *,
        tooltip: str = '',
        group: str = '',
    ) -> None:
        viewController = SpinBoxParameterViewController(parameter)
        self.addViewController(viewController, label, tooltip=tooltip, group=group)

    def addDecimalLineEdit(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tooltip: str = '',
        group: str = '',
    ) -> None:
        viewController = DecimalLineEditParameterViewController(parameter)
        self.addViewController(viewController, label, tooltip=tooltip, group=group)

    def addDecimalSlider(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tooltip: str = '',
        group: str = '',
    ) -> None:
        viewController = DecimalSliderParameterViewController(parameter)
        self.addViewController(viewController, label, tooltip=tooltip, group=group)

    def addLengthWidget(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tooltip: str = '',
        group: str = '',
    ) -> None:
        viewController = LengthWidgetParameterViewController(parameter)
        self.addViewController(viewController, label, tooltip=tooltip, group=group)

    def addAngleWidget(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tooltip: str = '',
        group: str = '',
    ) -> None:
        viewController = AngleWidgetParameterViewController(parameter)
        self.addViewController(viewController, label, tooltip=tooltip, group=group)

    def addViewControllerToTop(self, viewController: ParameterViewController) -> None:
        self._viewControllersTop.append(viewController)

    def addViewController(
        self,
        viewController: ParameterViewController,
        label: str,
        *,
        tooltip: str = '',
        group: str = '',
    ) -> None:
        self._viewControllers[group, label] = viewController

    def addViewControllerToBottom(self, viewController: ParameterViewController) -> None:
        self._viewControllersBottom.append(viewController)

    def _buildLayout(self) -> QVBoxLayout:
        groupDict: dict[str, QFormLayout] = dict()

        for (groupName, widgetLabel), vc in self._viewControllers.items():
            try:
                formLayout = groupDict[groupName]
            except KeyError:
                formLayout = QFormLayout()
                groupDict[groupName] = formLayout

            formLayout.addRow(widgetLabel, vc.getWidget())

        layout = QVBoxLayout()

        for viewController in self._viewControllersTop:
            layout.addWidget(viewController.getWidget())

        for groupName, groupLayout in groupDict.items():
            if groupName:
                groupBox = QGroupBox(groupName)
                groupBox.setLayout(groupLayout)
                layout.addWidget(groupBox)
            elif groupLayout.count() > 0:
                layout.addLayout(groupLayout)

        for viewController in self._viewControllersBottom:
            layout.addWidget(viewController.getWidget())

        return layout

    def _flushViewControllers(self) -> Sequence[ParameterViewController]:
        viewControllers: list[ParameterViewController] = list()
        viewControllers.extend(self._viewControllersTop)
        viewControllers.extend(self._viewControllers.values())
        viewControllers.extend(self._viewControllersBottom)

        self._viewControllersTop.clear()
        self._viewControllers.clear()
        self._viewControllersBottom.clear()

        return viewControllers

    def buildWidget(self) -> QWidget:
        layout = self._buildLayout()

        widget = ParameterWidget(self._flushViewControllers())
        widget.setLayout(layout)

        return widget

    def buildDialog(self, windowTitle: str, parent: QWidget | None) -> QDialog:
        buttonBox = QDialogButtonBox()
        layout = self._buildLayout()
        layout.addWidget(buttonBox)

        dialog = ParameterDialog(self._flushViewControllers(), buttonBox, parent)
        dialog.setLayout(layout)
        dialog.setWindowTitle(windowTitle)

        return dialog
