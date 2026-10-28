from PyQt5.QtWidgets import QDialog, QFormLayout, QGroupBox, QMessageBox, QSpinBox, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import BooleanParameter

from ...model.product.object import (
    FromFileObjectBuilder,
    FromMemoryObjectBuilder,
    ObjectRepositoryItem,
    PluginObjectBuilder,
)
from ..make_editable import MakeEditableViewController
from ..parameters import (
    ParameterViewBuilder,
    ParameterViewController,
    SpinBoxParameterViewController,
    create_parameter_view_controller,
)


class MultisliceViewController(ParameterViewController, Observer):
    def __init__(self, item: ObjectRepositoryItem) -> None:
        super().__init__()
        self._item = item
        self._parameter = item.layer_spacing_m
        self._widget = QSpinBox()
        self._widget.setToolTip(
            'Reslicing preserves the object: layers are resampled onto the new spacing'
            ' rather than discarded. Setting this to 1 leaves a multislice object alone,'
            ' and a reconstruction result is reassigned every iteration so it keeps the'
            ' layer count it was solved with.'
        )

        self._sync_model_to_view()
        self._widget.valueChanged.connect(self._sync_view_to_model)
        self._parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_view_to_model(self, num_layers: int) -> None:
        self._item.set_num_layers(num_layers)

    def _sync_model_to_view(self) -> None:
        self._widget.blockSignals(True)
        self._widget.setRange(1, 99)
        self._widget.setValue(self._item.get_num_layers())
        self._widget.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._sync_model_to_view()


class DeclaredParametersViewController(ParameterViewController):
    """The extra padding, then the rows a generator declares, in one group."""

    def __init__(self, title: str, builder: PluginObjectBuilder) -> None:
        super().__init__()
        self._view_controllers: list[ParameterViewController] = [
            SpinBoxParameterViewController(builder.extra_padding_x),
            SpinBoxParameterViewController(builder.extra_padding_y),
        ]
        layout = QFormLayout()
        layout.addRow('Extra Padding X:', self._view_controllers[0].get_widget())
        layout.addRow('Extra Padding Y:', self._view_controllers[1].get_widget())

        for description in builder.describe_parameters():
            parameter = builder.parameters()[description.name]
            view_controller = create_parameter_view_controller(
                parameter,
                name=description.name,
                label=description.label,
                tool_tip=description.tool_tip,
                choices=description.choices,
                unit=description.unit,
                display_unit=description.display_unit,
            )

            if view_controller is not None:
                self._view_controllers.append(view_controller)
                # A check box carries its own text, so its row needs no label beside it.
                row_label = '' if isinstance(parameter, BooleanParameter) else description.label
                layout.addRow(row_label, view_controller.get_widget())

        self._widget = QGroupBox(title)
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class ObjectEditorViewControllerFactory:
    def create_editor_dialog(
        self, item_name: str, item: ObjectRepositoryItem, parent: QWidget
    ) -> QDialog:
        object_builder = item.get_builder()
        builder_name = object_builder.get_name()
        additional_layers_group = 'Additional Layers'
        title = f'{item_name} [{builder_name}]'

        if isinstance(object_builder, PluginObjectBuilder):
            # One branch for every object model, because a generator describes its own
            # parameters: a model added later needs no branch of its own.
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_view_controller_to_top(
                DeclaredParametersViewController('First Layer', object_builder)
            )
            dialog_builder.add_view_controller(
                MultisliceViewController(item),
                'Number of Layers:',
                group=additional_layers_group,
            )
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(object_builder, FromFileObjectBuilder) or (
            isinstance(object_builder, FromMemoryObjectBuilder) and object_builder.is_editable
        ):
            # A from-file object is conditioned on the way in, and an editable result on
            # every build, but only the layer spacing applies -- the extra padding is
            # generation-only, so its spin boxes are deliberately absent here.
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_view_controller(
                MultisliceViewController(item),
                'Number of Layers:',
                group=additional_layers_group,
            )
            return dialog_builder.build_dialog(title, parent)
        elif item.can_make_editable():
            # A result ignores the layer spacing until it is made editable.
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_view_controller_to_bottom(
                MakeEditableViewController(
                    item.make_editable,
                    lambda: self.create_editor_dialog(item_name, item, parent).open(),
                    'Let this result be resliced into more layers. The spacing starts at '
                    'the one it has, so nothing changes until the layers are changed.',
                )
            )
            return dialog_builder.build_dialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builder_name}" has no editable parameters!',
            QMessageBox.StandardButton.Ok,
            parent,
        )
