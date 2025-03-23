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
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import StringParameter

from ...model.product.probe import (
    AveragePatternProbeBuilder,
    DiskProbeBuilder,
    FresnelZonePlateProbeBuilder,
    MultimodalProbeBuilder,
    ProbeModeDecayType,
    ProbeRepositoryItem,
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
from .zernike import ZernikeTableModel

__all__ = [
    'ProbeEditorViewControllerFactory',
]


class FresnelZonePlateViewController(ParameterViewController):
    def __init__(self, title: str, probe_builder: FresnelZonePlateProbeBuilder) -> None:
        super().__init__()
        self._widget = GroupBoxWithPresets(title)

        for label in probe_builder.labelsForPresets():
            action = self._widget.presets_menu.addAction(label)
            action.triggered.connect(lambda _, label=label: probe_builder.applyPresets(label))

        self._zonePlateDiameterViewController = LengthWidgetParameterViewController(
            probe_builder.zonePlateDiameterInMeters
        )
        self._outermostZoneWidthInMetersViewController = LengthWidgetParameterViewController(
            probe_builder.outermostZoneWidthInMeters
        )
        self._centralBeamstopDiameterInMetersViewController = LengthWidgetParameterViewController(
            probe_builder.centralBeamstopDiameterInMeters
        )
        self._defocusDistanceInMetersViewController = LengthWidgetParameterViewController(
            probe_builder.defocusDistanceInMeters
        )

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow('Zone Plate Diameter:', self._zonePlateDiameterViewController.get_widget())
        layout.addRow(
            'Outermost Zone Width:',
            self._outermostZoneWidthInMetersViewController.get_widget(),
        )
        layout.addRow(
            'Central Beamstop Diameter:',
            self._centralBeamstopDiameterInMetersViewController.get_widget(),
        )
        layout.addRow('Defocus Distance:', self._defocusDistanceInMetersViewController.get_widget())
        self._widget.contents.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class ZernikeViewController(ParameterViewController, Observer):
    def __init__(self, title: str, probe_builder: ZernikeProbeBuilder) -> None:
        super().__init__()
        self._widget = QGroupBox(title)
        self._probe_builder = probe_builder
        self._orderSpinBox = QSpinBox()
        self._coefficientsTableModel = ZernikeTableModel(probe_builder)
        self._coefficientsTableView = QTableView()
        self._diameterViewController = LengthWidgetParameterViewController(
            probe_builder.diameterInMeters
        )

        self._coefficientsTableView.setModel(self._coefficientsTableModel)

        layout = QFormLayout()
        layout.addRow('Diameter:', self._diameterViewController.get_widget())
        layout.addRow('Order:', self._orderSpinBox)
        layout.addRow(self._coefficientsTableView)
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        self._orderSpinBox.valueChanged.connect(probe_builder.setOrder)
        probe_builder.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._orderSpinBox.setRange(1, 100)
        self._orderSpinBox.setValue(self._probe_builder.getOrder())

        self._coefficientsTableModel.beginResetModel()  # TODO clean up
        self._coefficientsTableModel.endResetModel()

    def _update(self, observable: Observable) -> None:
        if observable is self._probe_builder:
            self._sync_model_to_view()


class DecayTypeParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: StringParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._polynomialDecayButton = QRadioButton('Polynomial')
        self._exponentialDecayButton = QRadioButton('Exponential')

        self._buttonGroup = QButtonGroup()
        self._buttonGroup.addButton(
            self._polynomialDecayButton, ProbeModeDecayType.POLYNOMIAL.value
        )
        self._buttonGroup.addButton(
            self._exponentialDecayButton, ProbeModeDecayType.EXPONENTIAL.value
        )
        self._buttonGroup.setExclusive(True)
        self._buttonGroup.idToggled.connect(self._sync_view_to_model)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._polynomialDecayButton)
        layout.addWidget(self._exponentialDecayButton)

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

        button = self._buttonGroup.button(decay_type.value)
        button.setChecked(True)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._sync_model_to_view()


class ProbeEditorViewControllerFactory:
    def _append_additional_modes(
        self,
        dialog_builder: ParameterViewBuilder,
        additional_modes_builder: MultimodalProbeBuilder | None,
    ) -> None:
        if additional_modes_builder is None:
            return

        additional_modes_group = 'Additional Modes'  # FIXME OPR
        dialog_builder.add_spin_box(
            additional_modes_builder.num_incoherent_modes,
            'Number of Modes:',
            group=additional_modes_group,
        )
        dialog_builder.add_check_box(
            additional_modes_builder.orthogonalize_incoherent_modes,
            'Orthogonalize Modes:',
            group=additional_modes_group,
        )
        dialog_builder.add_view_controller(
            DecayTypeParameterViewController(additional_modes_builder.incoherent_mode_decay_type),
            'Decay Type:',
            group=additional_modes_group,
        )
        dialog_builder.add_decimal_slider(
            additional_modes_builder.incoherent_mode_decay_ratio,
            'Decay Ratio:',
            group=additional_modes_group,
        )

    def create_editor_dialog(
        self, item_name: str, item: ProbeRepositoryItem, parent: QWidget
    ) -> QDialog:
        probe_builder = item.get_builder()
        builder_name = probe_builder.get_name()
        additional_modes_builder = item.get_additional_modes_builder()
        primary_mode_group = 'Primary Mode'
        title = f'{item_name} [{builder_name}]'

        if isinstance(probe_builder, AveragePatternProbeBuilder):
            dialog_builder = ParameterViewBuilder()
            self._append_additional_modes(dialog_builder, additional_modes_builder)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_builder, DiskProbeBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.addLengthWidget(
                probe_builder.diameterInMeters,
                'Diameter:',
                group=primary_mode_group,
            )
            dialog_builder.addLengthWidget(
                probe_builder.defocusDistanceInMeters,
                'Defocus Distance:',
                group=primary_mode_group,
            )
            self._append_additional_modes(dialog_builder, additional_modes_builder)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_builder, FresnelZonePlateProbeBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_view_controller_to_top(
                FresnelZonePlateViewController(primary_mode_group, probe_builder)
            )
            self._append_additional_modes(dialog_builder, additional_modes_builder)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_builder, RectangularProbeBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.addLengthWidget(
                probe_builder.widthInMeters,
                'Width:',
                group=primary_mode_group,
            )
            dialog_builder.addLengthWidget(
                probe_builder.heightInMeters,
                'Height:',
                group=primary_mode_group,
            )
            dialog_builder.addLengthWidget(
                probe_builder.defocusDistanceInMeters,
                'Defocus Distance:',
                group=primary_mode_group,
            )
            self._append_additional_modes(dialog_builder, additional_modes_builder)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_builder, SuperGaussianProbeBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.addLengthWidget(
                probe_builder.annularRadiusInMeters,
                'Annular Radius:',
                group=primary_mode_group,
            )
            dialog_builder.addLengthWidget(
                probe_builder.fwhmInMeters,
                'Full Width at Half Maximum:',
                group=primary_mode_group,
            )
            dialog_builder.add_decimal_line_edit(
                probe_builder.orderParameter,
                'Order Parameter:',
                group=primary_mode_group,
            )
            self._append_additional_modes(dialog_builder, additional_modes_builder)
            return dialog_builder.build_dialog(title, parent)
        elif isinstance(probe_builder, ZernikeProbeBuilder):
            dialog_builder = ParameterViewBuilder()
            dialog_builder.add_view_controller_to_top(
                ZernikeViewController(primary_mode_group, probe_builder)
            )
            self._append_additional_modes(dialog_builder, additional_modes_builder)
            return dialog_builder.build_dialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builder_name}" has no editable parameters!',
            QMessageBox.Ok,
            parent,
        )
