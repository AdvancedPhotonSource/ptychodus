from PyQt5.QtWidgets import QDialog, QMessageBox, QSpinBox, QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.product.object import ObjectRepositoryItem, RandomObjectBuilder
from ..parametric import ParameterViewBuilder, ParameterViewController


class MultisliceViewController(ParameterViewController, Observer):
    def __init__(self, item: ObjectRepositoryItem) -> None:
        super().__init__()
        self._item = item
        self._parameter = item.layer_spacing_m
        self._widget = QSpinBox()

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


class ObjectEditorViewControllerFactory:
    def create_editor_dialog(
        self, item_name: str, item: ObjectRepositoryItem, parent: QWidget
    ) -> QDialog:
        object_builder = item.get_builder()
        builder_name = object_builder.get_name()
        first_layer_group = 'First Layer'
        additional_layers_group = 'Additional Layers'
        title = f'{item_name} [{builder_name}]'

        if isinstance(object_builder, RandomObjectBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_spin_box(
                object_builder.extra_padding_x, 'Extra Padding X:', group=first_layer_group
            )
            dialog_builder.add_spin_box(
                object_builder.extra_padding_y, 'Extra Padding Y:', group=first_layer_group
            )
            dialog_builder.add_decimal_slider(
                object_builder.amplitude_mean, 'Amplitude Mean:', group=first_layer_group
            )
            dialog_builder.add_decimal_slider(
                object_builder.amplitude_deviation,
                'Amplitude Deviation:',
                group=first_layer_group,
            )
            dialog_builder.add_decimal_slider(
                object_builder.phase_deviation, 'Phase Deviation:', group=first_layer_group
            )
            dialog_builder.add_view_controller(
                MultisliceViewController(item),
                'Number of Layers:',
                group=additional_layers_group,
            )
            return dialog_builder.build_dialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builder_name}" has no editable parameters!',
            QMessageBox.StandardButton.Ok,
            parent,
        )
