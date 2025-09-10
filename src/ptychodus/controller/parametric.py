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
    def get_widget(self) -> QWidget:
        pass


class CheckableGroupBoxParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: BooleanParameter, title: str, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QGroupBox(title)
        self._widget.setCheckable(True)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.toggled.connect(parameter.set_value)
        self._parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_model_to_view(self) -> None:
        self._widget.setChecked(self._parameter.get_value())

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class CheckBoxParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: BooleanParameter, text: str, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QCheckBox(text)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.toggled.connect(parameter.set_value)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_model_to_view(self) -> None:
        self._widget.setChecked(self._parameter.get_value())

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class SpinBoxParameterViewController(ParameterViewController, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, parameter: IntegerParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QSpinBox()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.valueChanged.connect(parameter.set_value)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_model_to_view(self) -> None:
        minimum = self._parameter.get_minimum()
        maximum = self._parameter.get_maximum()

        if minimum is None:
            raise ValueError('Minimum not provided!')
        else:
            self._widget.blockSignals(True)

            if maximum is None:
                self._widget.setRange(minimum, SpinBoxParameterViewController.MAX_INT)
            else:
                self._widget.setRange(minimum, maximum)

            self._widget.setValue(self._parameter.get_value())
            self._widget.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


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

        self.__sync_model_to_view()
        self._widget.textActivated.connect(parameter.set_value)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_model_to_view(self) -> None:
        self._widget.setCurrentText(self._parameter.get_value())

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


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

        self.__sync_model_to_view()
        self._widget.editingFinished.connect(self.__sync_view_to_model)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_view_to_model(self) -> None:
        self._parameter.set_value(self._widget.text())

    def __sync_model_to_view(self) -> None:
        self._widget.setText(self._parameter.get_value())

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class PathParameterViewController(ParameterViewController, Observer):
    def __init__(
        self,
        parameter: PathParameter,
        file_dialog_factory: FileDialogFactory,
        *,
        caption: str,
        name_filters: Sequence[str] | None,
        mime_type_filters: Sequence[str] | None,
        selected_name_filter: str | None,
        tool_tip: str,
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._file_dialog_factory = file_dialog_factory
        self._caption = caption
        self._name_filters = name_filters
        self._mime_type_filters = mime_type_filters
        self._selected_name_filter = selected_name_filter
        self._line_edit = QLineEdit()
        self._browse_button = QPushButton('Browse')
        self._widget = QWidget()

        if tool_tip:
            self._line_edit.setToolTip(tool_tip)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._line_edit)
        layout.addWidget(self._browse_button)
        self._widget.setLayout(layout)

        self.__sync_model_to_view()
        parameter.add_observer(self)
        self._line_edit.editingFinished.connect(self.__sync_path_to_model)

    @classmethod
    def create_file_opener(
        cls,
        parameter: PathParameter,
        file_dialog_factory: FileDialogFactory,
        *,
        caption: str = 'Open File',
        name_filters: Sequence[str] | None = None,
        mime_type_filters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
        tool_tip: str = '',
    ) -> PathParameterViewController:
        view_controller = cls(
            parameter,
            file_dialog_factory,
            caption=caption,
            name_filters=name_filters,
            mime_type_filters=mime_type_filters,
            selected_name_filter=selected_name_filter,
            tool_tip=tool_tip,
        )
        view_controller._browse_button.clicked.connect(view_controller._choose_file_to_open)
        return view_controller

    @classmethod
    def create_file_saver(
        cls,
        parameter: PathParameter,
        file_dialog_factory: FileDialogFactory,
        *,
        caption: str = 'Save File',
        name_filters: Sequence[str] | None = None,
        mime_type_filters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
        tool_tip: str = '',
    ) -> PathParameterViewController:
        view_controller = cls(
            parameter,
            file_dialog_factory,
            caption=caption,
            name_filters=name_filters,
            mime_type_filters=mime_type_filters,
            selected_name_filter=selected_name_filter,
            tool_tip=tool_tip,
        )
        view_controller._browse_button.clicked.connect(view_controller._choose_file_to_save)
        return view_controller

    @classmethod
    def create_directory_chooser(
        cls,
        parameter: PathParameter,
        file_dialog_factory: FileDialogFactory,
        *,
        caption: str = 'Choose Directory',
        tool_tip: str = '',
    ) -> PathParameterViewController:
        view_controller = cls(
            parameter,
            file_dialog_factory,
            caption=caption,
            name_filters=None,
            mime_type_filters=None,
            selected_name_filter=None,
            tool_tip=tool_tip,
        )
        view_controller._browse_button.clicked.connect(view_controller._choose_directory)
        return view_controller

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_path_to_model(self) -> None:
        path = Path(self._line_edit.text())
        self._parameter.set_value(path)

    def _choose_file_to_open(self) -> None:
        path, _ = self._file_dialog_factory.get_open_file_path(
            self._widget,
            self._caption,
            self._name_filters,
            self._mime_type_filters,
            self._selected_name_filter,
        )

        if path:
            self._parameter.set_value(path)

    def _choose_file_to_save(self) -> None:
        path, _ = self._file_dialog_factory.get_save_file_path(
            self._widget,
            self._caption,
            self._name_filters,
            self._mime_type_filters,
            self._selected_name_filter,
        )

        if path:
            self._parameter.set_value(path)

    def _choose_directory(self) -> None:
        path = self._file_dialog_factory.get_existing_directory_path(self._widget, self._caption)

        if path:
            self._parameter.set_value(path)

    def __sync_model_to_view(self) -> None:
        path = self._parameter.get_value()
        self._line_edit.setText(str(path))

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class IntegerLineEditParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: IntegerParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QLineEdit()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        validator = QIntValidator()
        bottom = parameter.get_minimum()
        top = parameter.get_maximum()

        if bottom is not None:
            validator.setBottom(bottom)

        if top is not None:
            validator.setTop(top)

        self._widget.setValidator(validator)

        self.__sync_model_to_view()
        self._widget.editingFinished.connect(self.__sync_view_to_model)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_view_to_model(self) -> None:
        text = self._widget.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert "{text}" to int!')
        else:
            self._parameter.set_value(value)

    def __sync_model_to_view(self) -> None:
        self._widget.setText(str(self._parameter.get_value()))

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class DecimalLineEditParameterViewController(ParameterViewController, Observer):
    def __init__(
        self, parameter: RealParameter, *, is_signed: bool = False, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalLineEdit.create_instance(is_signed=is_signed)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.value_changed.connect(self.__sync_view_to_model)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_view_to_model(self, value: Decimal) -> None:
        self._parameter.set_value(float(value))

    def __sync_model_to_view(self) -> None:
        self._widget.set_value(Decimal(str(self._parameter.get_value())))

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class DecimalSliderParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: RealParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = DecimalSlider.create_instance(Qt.Orientation.Horizontal)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.value_changed.connect(self.__sync_view_to_model)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_view_to_model(self, value: Decimal) -> None:
        self._parameter.set_value(float(value))

    def __sync_model_to_view(self) -> None:
        minimum = self._parameter.get_minimum()
        maximum = self._parameter.get_maximum()

        if minimum is None or maximum is None:
            raise ValueError('Range not provided!')
        else:
            value = Decimal(str(self._parameter.get_value()))
            range_ = Interval[Decimal](Decimal(str(minimum)), Decimal(str(maximum)))
            self._widget.set_value_and_range(value, range_)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class LengthWidgetParameterViewController(ParameterViewController, Observer):
    def __init__(
        self, parameter: RealParameter, *, is_signed: bool = False, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = LengthWidget(is_signed=is_signed)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.length_changed.connect(self.__sync_view_to_model)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_view_to_model(self, value: Decimal) -> None:
        self._parameter.set_value(float(value))

    def __sync_model_to_view(self) -> None:
        self._widget.set_length_m(Decimal(str(self._parameter.get_value())))

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class AngleWidgetParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: RealParameter, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = AngleWidget()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.angle_changed.connect(self.__sync_view_to_model)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_view_to_model(self, value: Decimal) -> None:
        self._parameter.set_value(float(value))

    def __sync_model_to_view(self) -> None:
        self._widget.set_angle_in_turns(Decimal(str(self._parameter.get_value())))

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class ParameterWidget(QWidget):
    def __init__(
        self, view_controllers: Sequence[ParameterViewController], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._view_controllers = view_controllers


class ParameterDialog(QDialog):
    def __init__(
        self,
        view_controllers: Sequence[ParameterViewController],
        button_box: QDialogButtonBox,
        parent: QWidget | None,
    ) -> None:
        super().__init__(parent)
        self._view_controllers = view_controllers
        self._button_box = button_box

        button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        button_box.clicked.connect(self._handle_button_box_clicked)

    def _handle_button_box_clicked(self, button: QAbstractButton) -> None:
        # TODO remove observers from viewControllers

        if self._button_box.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class ParameterViewBuilder:
    def __init__(self, file_dialog_factory: FileDialogFactory | None = None) -> None:
        self._file_dialog_factory = file_dialog_factory
        self._view_controllers_top: list[ParameterViewController] = list()
        self._view_controllers: dict[tuple[str, str], ParameterViewController] = dict()
        self._view_controllers_bottom: list[ParameterViewController] = list()

    def add_check_box(
        self,
        parameter: BooleanParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = CheckBoxParameterViewController(parameter, '', tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_combo_box(
        self,
        parameter: StringParameter,
        items: Iterable[str],
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = ComboBoxParameterViewController(parameter, items, tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_file_opener(
        self,
        parameter: PathParameter,
        label: str,
        *,
        caption: str = 'Open File',
        name_filters: Sequence[str] | None = None,
        mime_type_filters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        if self._file_dialog_factory is None:
            raise ValueError('Cannot add file chooser without FileDialogFactory!')
        else:
            view_controller = PathParameterViewController.create_file_opener(
                parameter,
                self._file_dialog_factory,
                caption=caption,
                name_filters=name_filters,
                mime_type_filters=mime_type_filters,
                selected_name_filter=selected_name_filter,
                tool_tip=tool_tip,
            )
            self.add_view_controller(view_controller, label, group=group)

    def add_file_saver(
        self,
        parameter: PathParameter,
        label: str,
        *,
        caption: str = 'Save File',
        name_filters: Sequence[str] | None = None,
        mime_type_filters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        if self._file_dialog_factory is None:
            raise ValueError('Cannot add file chooser without FileDialogFactory!')
        else:
            view_controller = PathParameterViewController.create_file_saver(
                parameter,
                self._file_dialog_factory,
                caption=caption,
                name_filters=name_filters,
                mime_type_filters=mime_type_filters,
                selected_name_filter=selected_name_filter,
                tool_tip=tool_tip,
            )
            self.add_view_controller(view_controller, label, group=group)

    def add_directory_chooser(
        self, parameter: PathParameter, label: str, *, tool_tip: str = '', group: str = ''
    ) -> None:
        if self._file_dialog_factory is None:
            raise ValueError('Cannot add directory chooser without FileDialogFactory!')
        else:
            view_controller = PathParameterViewController.create_directory_chooser(
                parameter,
                self._file_dialog_factory,
                tool_tip=tool_tip,
            )
            self.add_view_controller(view_controller, label, group=group)

    def add_spin_box(
        self,
        parameter: IntegerParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = SpinBoxParameterViewController(parameter, tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_integer_line_edit(
        self,
        parameter: IntegerParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = IntegerLineEditParameterViewController(parameter, tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_decimal_line_edit(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = DecimalLineEditParameterViewController(parameter, tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_decimal_slider(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = DecimalSliderParameterViewController(parameter, tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_length_widget(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = LengthWidgetParameterViewController(parameter, tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_angle_widget(
        self,
        parameter: RealParameter,
        label: str,
        *,
        tool_tip: str = '',
        group: str = '',
    ) -> None:
        view_controller = AngleWidgetParameterViewController(parameter, tool_tip=tool_tip)
        self.add_view_controller(view_controller, label, group=group)

    def add_view_controller_to_top(self, view_controller: ParameterViewController) -> None:
        self._view_controllers_top.append(view_controller)

    def add_view_controller(
        self,
        view_controller: ParameterViewController,
        label: str,
        *,
        group: str = '',
    ) -> None:
        self._view_controllers[group, label] = view_controller

    def add_view_controller_to_bottom(self, view_controller: ParameterViewController) -> None:
        self._view_controllers_bottom.append(view_controller)

    def _build_layout(self, *, add_stretch: bool) -> QVBoxLayout:
        group_dict: dict[str, QFormLayout] = dict()

        for (group_name, widget_label), vc in self._view_controllers.items():
            try:
                form_layout = group_dict[group_name]
            except KeyError:
                form_layout = QFormLayout()
                group_dict[group_name] = form_layout

            form_layout.addRow(widget_label, vc.get_widget())

        layout = QVBoxLayout()

        for view_controller in self._view_controllers_top:
            layout.addWidget(view_controller.get_widget())

        for group_name, group_layout in group_dict.items():
            if group_name:
                group_box = QGroupBox(group_name)
                group_box.setLayout(group_layout)
                layout.addWidget(group_box)
            elif group_layout.count() > 0:
                layout.addLayout(group_layout)

        for view_controller in self._view_controllers_bottom:
            layout.addWidget(view_controller.get_widget())

        if add_stretch:
            layout.addStretch()

        return layout

    def _flush_view_controllers(self) -> Sequence[ParameterViewController]:
        view_controllers: list[ParameterViewController] = list()
        view_controllers.extend(self._view_controllers_top)
        view_controllers.extend(self._view_controllers.values())
        view_controllers.extend(self._view_controllers_bottom)

        self._view_controllers_top.clear()
        self._view_controllers.clear()
        self._view_controllers_bottom.clear()

        return view_controllers

    def build_widget(self) -> QWidget:
        layout = self._build_layout(add_stretch=True)

        widget = ParameterWidget(self._flush_view_controllers())
        widget.setLayout(layout)

        return widget

    def build_dialog(self, window_title: str, parent: QWidget | None) -> QDialog:
        button_box = QDialogButtonBox()
        layout = self._build_layout(add_stretch=True)
        layout.addWidget(button_box)

        dialog = ParameterDialog(self._flush_view_controllers(), button_box, parent)
        dialog.setLayout(layout)
        dialog.setWindowTitle(window_title)

        return dialog
