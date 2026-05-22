from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import StringParameter

from ...model.product.probe import (
    AveragePatternProbeBuilder,
    DiskProbeBuilder,
    FresnelZonePlateProbeBuilder,
    HermiteProbeBuilder,
    ProbeModeDecayType,
    ProbeRepositoryItem,
    ProbeSequenceBuilder,
    RectangularProbeBuilder,
    SuperGaussianProbeBuilder,
    ZernikeProbeBuilder,
)
from ...view.widgets import GroupBoxWithPresets
from ..parametric import (
    LengthWidgetParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
)
from .hermite import HermiteTableModel
from .metrics import ProbeMetricsTableModel
from .zernike import ZernikeTableModel

__all__ = [
    'ProbeEditorViewControllerFactory',
]


class FresnelZonePlateViewController(ParameterViewController):
    def __init__(self, title: str, probe_builder: FresnelZonePlateProbeBuilder) -> None:
        super().__init__()
        self._widget = GroupBoxWithPresets(title)

        for label in probe_builder.labels_for_presets():
            action = self._widget.presets_menu.addAction(label)

            if action is None:
                raise ValueError('action is None!')
            else:
                action.triggered.connect(lambda _, label=label: probe_builder.apply_presets(label))

        self._zone_plate_diameter_view_controller = LengthWidgetParameterViewController(
            probe_builder.zone_plate_diameter_m
        )
        self._outermost_zone_width_view_controller = LengthWidgetParameterViewController(
            probe_builder.outermost_zone_width_m
        )
        self._central_beamstop_diameter_view_controller = LengthWidgetParameterViewController(
            probe_builder.central_beamstop_diameter_m
        )
        self._defocus_distance_view_controller = LengthWidgetParameterViewController(
            probe_builder.defocus_distance_m
        )

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(
            'Zone Plate Diameter:', self._zone_plate_diameter_view_controller.get_widget()
        )
        layout.addRow(
            'Outermost Zone Width:',
            self._outermost_zone_width_view_controller.get_widget(),
        )
        layout.addRow(
            'Central Beamstop Diameter:',
            self._central_beamstop_diameter_view_controller.get_widget(),
        )
        layout.addRow('Defocus Distance:', self._defocus_distance_view_controller.get_widget())
        self._widget.contents.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class ZernikeViewController(ParameterViewController, Observer):
    def __init__(self, title: str, probe_builder: ZernikeProbeBuilder) -> None:
        super().__init__()
        self._widget = QGroupBox(title)
        self._probe_builder = probe_builder
        self._order_spin_box = QSpinBox()
        self._coefficients_table_model = ZernikeTableModel(probe_builder)
        self._coefficients_table_view = QTableView()
        self._diameter_view_controller = LengthWidgetParameterViewController(
            probe_builder.diameter_m
        )

        self._coefficients_table_view.setModel(self._coefficients_table_model)
        header = self._coefficients_table_view.horizontalHeader()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)

        layout = QFormLayout()
        layout.addRow('Diameter:', self._diameter_view_controller.get_widget())
        layout.addRow('Order:', self._order_spin_box)
        layout.addRow(self._coefficients_table_view)
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        self._order_spin_box.valueChanged.connect(probe_builder.set_order)
        probe_builder.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._order_spin_box.setRange(1, 100)
        self._order_spin_box.setValue(self._probe_builder.get_order())

        self._coefficients_table_model.beginResetModel()  # TODO clean up
        self._coefficients_table_model.endResetModel()

    def _update(self, observable: Observable) -> None:
        if observable is self._probe_builder:
            self._sync_model_to_view()


class HermiteViewController(ParameterViewController, Observer):
    def __init__(self, title: str, probe_builder: HermiteProbeBuilder) -> None:
        super().__init__()
        self._widget = QGroupBox(title)
        self._probe_builder = probe_builder
        self._order_x_spin_box = QSpinBox()
        self._order_y_spin_box = QSpinBox()
        self._coefficients_table_model = HermiteTableModel(probe_builder)
        self._coefficients_table_view = QTableView()
        self._width_view_controller = LengthWidgetParameterViewController(probe_builder.width_m)
        self._height_view_controller = LengthWidgetParameterViewController(probe_builder.height_m)

        self._coefficients_table_view.setModel(self._coefficients_table_model)
        header = self._coefficients_table_view.horizontalHeader()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)

        layout = QFormLayout()
        layout.addRow('Width:', self._width_view_controller.get_widget())
        layout.addRow('Height:', self._height_view_controller.get_widget())
        layout.addRow('Order X:', self._order_x_spin_box)
        layout.addRow('Order Y:', self._order_y_spin_box)
        layout.addRow(self._coefficients_table_view)
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        self._order_x_spin_box.valueChanged.connect(probe_builder.set_order_x)
        self._order_y_spin_box.valueChanged.connect(probe_builder.set_order_y)
        probe_builder.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._order_x_spin_box.setRange(1, 100)
        self._order_x_spin_box.setValue(self._probe_builder.get_order_x())
        self._order_y_spin_box.setRange(1, 100)
        self._order_y_spin_box.setValue(self._probe_builder.get_order_y())

        self._coefficients_table_model.beginResetModel()  # TODO clean up
        self._coefficients_table_model.endResetModel()

    def _update(self, observable: Observable) -> None:
        if observable is self._probe_builder:
            self._sync_model_to_view()


class DecayTypeParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: StringParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._polynomial_decay_button = QRadioButton('Polynomial')
        self._exponential_decay_button = QRadioButton('Exponential')

        self._button_group = QButtonGroup()
        self._button_group.addButton(
            self._polynomial_decay_button, ProbeModeDecayType.POLYNOMIAL.value
        )
        self._button_group.addButton(
            self._exponential_decay_button, ProbeModeDecayType.EXPONENTIAL.value
        )
        self._button_group.setExclusive(True)
        self._button_group.idToggled.connect(self._sync_view_to_model)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._polynomial_decay_button)
        layout.addWidget(self._exponential_decay_button)

        self._widget = QWidget()
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_view_to_model(self, tool_id: int, checked: bool) -> None:
        if checked:
            decay_type = ProbeModeDecayType(tool_id)
            self._parameter.set_value(decay_type.name)

    def _sync_model_to_view(self) -> None:
        try:
            decay_type = ProbeModeDecayType[self._parameter.get_value().upper()]
        except KeyError:
            decay_type = ProbeModeDecayType.POLYNOMIAL

        button = self._button_group.button(decay_type.value)

        if button is None:
            raise ValueError('button is None!')
        else:
            button.setChecked(True)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._sync_model_to_view()


class ProbeMetricsViewController(ParameterViewController, Observer):
    def __init__(self, title: str, probe_item: ProbeRepositoryItem) -> None:
        super().__init__()
        self._probe_item = probe_item
        self._widget = QGroupBox(title)
        self._table_model = ProbeMetricsTableModel()
        self._table_view = QTableView()
        self._table_view.setModel(self._table_model)

        vertical_header = self._table_view.verticalHeader()
        if vertical_header is not None:
            vertical_header.hide()

        header = self._table_view.horizontalHeader()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)

        layout = QVBoxLayout()
        layout.addWidget(self._table_view)
        self._widget.setLayout(layout)

        self._refresh_metrics()
        probe_item.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _refresh_metrics(self) -> None:
        metrics = self._probe_item.get_size_metrics()
        self._table_model.set_metrics(metrics)
        self._table_view.resizeRowsToContents()

    def _update(self, observable: Observable) -> None:
        if observable is self._probe_item:
            self._refresh_metrics()


class ProbeEditorViewControllerFactory:
    def _append_primary_mode(
        self,
        probe_builder: ProbeSequenceBuilder,
        dialog_builder: ParameterViewBuilder,
    ) -> bool:
        primary_mode_group = 'Primary Mode'

        if isinstance(probe_builder, AveragePatternProbeBuilder):
            return True
        elif isinstance(probe_builder, DiskProbeBuilder):
            dialog_builder.add_length_widget(
                probe_builder.diameter_m,
                'Diameter:',
                group=primary_mode_group,
            )
            dialog_builder.add_length_widget(
                probe_builder.defocus_distance_m,
                'Defocus Distance:',
                group=primary_mode_group,
            )
            return True
        elif isinstance(probe_builder, FresnelZonePlateProbeBuilder):
            dialog_builder.add_view_controller_to_top(
                FresnelZonePlateViewController(primary_mode_group, probe_builder)
            )
            return True
        elif isinstance(probe_builder, HermiteProbeBuilder):
            dialog_builder.add_view_controller_to_top(
                HermiteViewController(primary_mode_group, probe_builder)
            )
            return True
        elif isinstance(probe_builder, RectangularProbeBuilder):
            dialog_builder.add_length_widget(
                probe_builder.width_m,
                'Width:',
                group=primary_mode_group,
            )
            dialog_builder.add_length_widget(
                probe_builder.height_m,
                'Height:',
                group=primary_mode_group,
            )
            dialog_builder.add_length_widget(
                probe_builder.defocus_distance_m,
                'Defocus Distance:',
                group=primary_mode_group,
            )
            return True
        elif isinstance(probe_builder, SuperGaussianProbeBuilder):
            dialog_builder.add_length_widget(
                probe_builder.annular_radius_m,
                'Annular Radius:',
                group=primary_mode_group,
            )
            dialog_builder.add_length_widget(
                probe_builder.fwhm_m,
                'Full Width at Half Maximum:',
                group=primary_mode_group,
            )
            dialog_builder.add_decimal_line_edit(
                probe_builder.order_parameter,
                'Order Parameter:',
                group=primary_mode_group,
            )
            return True
        elif isinstance(probe_builder, ZernikeProbeBuilder):
            dialog_builder.add_view_controller_to_top(
                ZernikeViewController(primary_mode_group, probe_builder)
            )
            return True

        return False

    def _append_additional_modes(
        self,
        probe_builder: ProbeSequenceBuilder,
        dialog_builder: ParameterViewBuilder,
    ) -> None:
        incoherent_modes_group = 'Incoherent (Mixed State) Modes'
        dialog_builder.add_spin_box(
            probe_builder.num_incoherent_modes,
            'Number of Modes:',
            group=incoherent_modes_group,
        )
        dialog_builder.add_check_box(
            probe_builder.orthogonalize_incoherent_modes,
            'Orthogonalize Modes:',
            group=incoherent_modes_group,
        )
        dialog_builder.add_view_controller(
            DecayTypeParameterViewController(probe_builder.incoherent_mode_decay_type),
            'Decay Type:',
            group=incoherent_modes_group,
        )
        dialog_builder.add_decimal_slider(
            probe_builder.incoherent_mode_decay_ratio,
            'Decay Ratio:',
            group=incoherent_modes_group,
        )

        coherent_modes_group = 'Coherent (OPR) Modes'
        dialog_builder.add_spin_box(
            probe_builder.num_coherent_modes,
            'Number of Modes:',
            group=coherent_modes_group,
        )

    def create_editor_dialog(
        self, item_name: str, item: ProbeRepositoryItem, parent: QWidget
    ) -> QDialog:
        probe_builder = item.get_builder()
        builder_name = probe_builder.get_name()
        title = f'{item_name} [{builder_name}]'

        dialog_builder = ParameterViewBuilder()
        dialog_builder.add_view_controller_to_top(ProbeMetricsViewController('Probe Metrics', item))

        if self._append_primary_mode(probe_builder, dialog_builder):
            self._append_additional_modes(probe_builder, dialog_builder)
            return dialog_builder.build_dialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builder_name}" has no editable parameters!',
            QMessageBox.StandardButton.Ok,
            parent,
        )
