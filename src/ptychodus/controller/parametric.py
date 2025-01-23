from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
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
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    PathParameter,
    RealParameter,
    StringParameter,
)

from ..view.widgets import AngleWidget, DecimalLineEdit, DecimalSlider, LengthWidget
from .data import FileDialogFactory

logger = logging.getLogger(__name__)


class ParameterViewController(ABC):
    @abstractmethod
    def getWidget(self) -> QWidget:
        pass


class CheckableGroupBoxParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: BooleanParameter, title: str, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QGroupBox(title)
        self._widget.setCheckable(True)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._syncModelToView()
        self._widget.toggled.connect(parameter.setValue)
        self._parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._parameter.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class CheckBoxParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: BooleanParameter, text: str, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QCheckBox(text)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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

    def __init__(self, parameter: IntegerParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QSpinBox()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._syncModelToView()
        self._widget.valueChanged.connect(parameter.setValue)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None:
            raise ValueError('Minimum not provided!')
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
        self, parameter: StringParameter, items: Iterable[str], *, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QComboBox()

        for item in items:
            self._widget.addItem(item)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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
        self, parameter: StringParameter, validator: QValidator | None = None, *, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QLineEdit()

        if validator is not None:
            self._widget.setValidator(validator)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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


class PathParameterViewController(ParameterViewController, Observer):
    def __init__(
        self,
        parameter: PathParameter,
        fileDialogFactory: FileDialogFactory,
        *,
        caption: str,
        nameFilters: Sequence[str] | None,
        mimeTypeFilters: Sequence[str] | None,
        selectedNameFilter: str | None,
        tool_tip: str,
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._fileDialogFactory = fileDialogFactory
        self._caption = caption
        self._nameFilters = nameFilters
        self._mimeTypeFilters = mimeTypeFilters
        self._selectedNameFilter = selectedNameFilter
        self._lineEdit = QLineEdit()
        self._browseButton = QPushButton('Browse')
        self._widget = QWidget()

        if tool_tip:
            self._lineEdit.setToolTip(tool_tip)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._lineEdit)
        layout.addWidget(self._browseButton)
        self._widget.setLayout(layout)

        self._syncModelToView()
        parameter.addObserver(self)
        self._lineEdit.editingFinished.connect(self._syncPathToModel)

    @classmethod
    def createFileOpener(
        cls,
        parameter: PathParameter,
        fileDialogFactory: FileDialogFactory,
        *,
        caption: str = 'Open File',
        nameFilters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selectedNameFilter: str | None = None,
        tool_tip: str = '',
    ) -> PathParameterViewController:
        viewController = cls(
            parameter,
            fileDialogFactory,
            caption=caption,
            nameFilters=nameFilters,
            mimeTypeFilters=mimeTypeFilters,
            selectedNameFilter=selectedNameFilter,
            tool_tip=tool_tip,
        )
        viewController._browseButton.clicked.connect(viewController._chooseFileToOpen)
        return viewController

    @classmethod
    def createFileSaver(
        cls,
        parameter: PathParameter,
        fileDialogFactory: FileDialogFactory,
        *,
        caption: str = 'Save File',
        nameFilters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selectedNameFilter: str | None = None,
        tool_tip: str = '',
    ) -> PathParameterViewController:
        viewController = cls(
            parameter,
            fileDialogFactory,
            caption=caption,
            nameFilters=nameFilters,
            mimeTypeFilters=mimeTypeFilters,
            selectedNameFilter=selectedNameFilter,
            tool_tip=tool_tip,
        )
        viewController._browseButton.clicked.connect(viewController._chooseFileToSave)
        return viewController

    @classmethod
    def createDirectoryChooser(
        cls,
        parameter: PathParameter,
        fileDialogFactory: FileDialogFactory,
        *,
        caption: str = 'Choose Directory',
        tool_tip: str = '',
    ) -> PathParameterViewController:
        viewController = cls(
            parameter,
            fileDialogFactory,
            caption=caption,
            nameFilters=None,
            mimeTypeFilters=None,
            selectedNameFilter=None,
            tool_tip=tool_tip,
        )
        viewController._browseButton.clicked.connect(viewController._chooseDirectory)
        return viewController

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncPathToModel(self) -> None:
        path = Path(self._lineEdit.text())
        self._parameter.setValue(path)

    def _chooseFileToOpen(self) -> None:
        path, _ = self._fileDialogFactory.getOpenFilePath(
            self._widget,
            self._caption,
            self._nameFilters,
            self._mimeTypeFilters,
            self._selectedNameFilter,
        )

        if path:
            self._parameter.setValue(path)

    def _chooseFileToSave(self) -> None:
        path, _ = self._fileDialogFactory.getSaveFilePath(
            self._widget,
            self._caption,
            self._nameFilters,
            self._mimeTypeFilters,
            self._selectedNameFilter,
        )

        if path:
            self._parameter.setValue(path)

    def _chooseDirectory(self) -> None:
        path = self._fileDialogFactory.getExistingDirectoryPath(self._widget, self._caption)

        if path:
            self._parameter.setValue(path)

    def _syncModelToView(self) -> None:
        path = self._parameter.getValue()
        self._lineEdit.setText(str(path))

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class IntegerLineEditParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: IntegerParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QLineEdit()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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
        self, parameter: RealParameter, *, is_signed: bool = False, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalLineEdit.createInstance(isSigned=is_signed)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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
    def __init__(self, parameter: RealParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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
            raise ValueError('Range not provided!')
        else:
            value = Decimal(repr(self._parameter.getValue()))
            range_ = Interval[Decimal](Decimal(repr(minimum)), Decimal(repr(maximum)))
            self._widget.setValueAndRange(value, range_)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class LengthWidgetParameterViewController(ParameterViewController, Observer):
    def __init__(
        self, parameter: RealParameter, *, is_signed: bool = False, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = LengthWidget.createInstance(isSigned=is_signed)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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
    def __init__(self, parameter: RealParameter, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = AngleWidget.createInstance()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

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
    def __init__(self, fileDialogFactory: FileDialogFactory | None = None) -> None:
        self._fileDialogFactory = fileDialogFactory
        self._viewControllersTop: list[ParameterViewController] = list()
        self._viewControllers: dict[tuple[str, str], ParameterViewController] = dict()
        self._viewControllersBottom: list[ParameterViewController] = list()

    def addCheckBox(
        self,
        parameter: BooleanParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = CheckBoxParameterViewController(parameter, '', tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addComboBox(
        self,
        parameter: StringParameter,
        items: Iterable[str],
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = ComboBoxParameterViewController(parameter, items, tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addFileOpener(
        self,
        parameter: PathParameter,
        label: str,
        *,
        caption: str = 'Open File',
        nameFilters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selectedNameFilter: str | None = None,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        if self._fileDialogFactory is None:
            raise ValueError('Cannot add file chooser without FileDialogFactory!')
        else:
            viewController = PathParameterViewController.createFileOpener(
                parameter,
                self._fileDialogFactory,
                caption=caption,
                nameFilters=nameFilters,
                mimeTypeFilters=mimeTypeFilters,
                selectedNameFilter=selectedNameFilter,
                tool_tip=tool_tip,
            )
            self.addViewController(viewController, label, group=group)

    def addFileSaver(
        self,
        parameter: PathParameter,
        label: str,
        *,
        caption: str = 'Save File',
        nameFilters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selectedNameFilter: str | None = None,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        if self._fileDialogFactory is None:
            raise ValueError('Cannot add file chooser without FileDialogFactory!')
        else:
            viewController = PathParameterViewController.createFileSaver(
                parameter,
                self._fileDialogFactory,
                caption=caption,
                nameFilters=nameFilters,
                mimeTypeFilters=mimeTypeFilters,
                selectedNameFilter=selectedNameFilter,
                tool_tip=tool_tip,
            )
            self.addViewController(viewController, label, group=group)

    def addDirectoryChooser(
        self, parameter: PathParameter, label: str, *, tool_tip: str = '', group: str = ''
    ) -> None:
        if self._fileDialogFactory is None:
            raise ValueError('Cannot add directory chooser without FileDialogFactory!')
        else:
            viewController = PathParameterViewController.createDirectoryChooser(
                parameter,
                self._fileDialogFactory,
                tool_tip=tool_tip,
            )
            self.addViewController(viewController, label, group=group)

    def addSpinBox(
        self,
        parameter: IntegerParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = SpinBoxParameterViewController(parameter, tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addIntegerLineEdit(
        self,
        parameter: IntegerParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = IntegerLineEditParameterViewController(parameter, tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addDecimalLineEdit(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = DecimalLineEditParameterViewController(parameter, tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addDecimalSlider(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = DecimalSliderParameterViewController(parameter, tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addLengthWidget(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = LengthWidgetParameterViewController(parameter, tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addAngleWidget(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        viewController = AngleWidgetParameterViewController(parameter, tool_tip=tool_tip)
        self.addViewController(viewController, label, group=group)

    def addViewControllerToTop(self, viewController: ParameterViewController) -> None:
        self._viewControllersTop.append(viewController)

    def addViewController(
        self,
        viewController: ParameterViewController,
        label: str,
        *,
        group: str = '',
    ) -> None:
        self._viewControllers[group, label] = viewController

    def addViewControllerToBottom(self, viewController: ParameterViewController) -> None:
        self._viewControllersBottom.append(viewController)

    def _buildLayout(self, *, add_stretch: bool) -> QVBoxLayout:
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

        if add_stretch:
            layout.addStretch()

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
        layout = self._buildLayout(add_stretch=True)

        widget = ParameterWidget(self._flushViewControllers())
        widget.setLayout(layout)

        return widget

    def buildDialog(self, windowTitle: str, parent: QWidget | None) -> QDialog:
        buttonBox = QDialogButtonBox()
        layout = self._buildLayout(add_stretch=False)
        layout.addWidget(buttonBox)

        dialog = ParameterDialog(self._flushViewControllers(), buttonBox, parent)
        dialog.setLayout(layout)
        dialog.setWindowTitle(windowTitle)

        return dialog
