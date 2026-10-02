from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.constants import LengthUnit
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe import OPRWeightPolicy
from ptychodus.api.parameters import StringParameter

from ...model.product.probe import (
    INCOHERENT_MODE_STRATEGY_NAMES,
    AveragePatternProbeBuilder,
    DiskProbeBuilder,
    FresnelZonePlateProbeBuilder,
    FromFileProbeBuilder,
    HermiteProbeBuilder,
    KBMirrorProbeBuilder,
    ProbeModeDecayType,
    ProbeRepositoryItem,
    ProbeSequenceBuilder,
    RectangularProbeBuilder,
    SuperGaussianProbeBuilder,
    ZernikeProbeBuilder,
)
from ...view.widgets import GroupBoxWithPresets
from ..parameters import (
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LengthParameterViewController,
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

        self._zone_plate_diameter_view_controller = LengthParameterViewController(
            probe_builder.zone_plate_diameter_m
        )
        self._outermost_zone_width_view_controller = LengthParameterViewController(
            probe_builder.outermost_zone_width_m
        )
        self._central_beamstop_diameter_view_controller = LengthParameterViewController(
            probe_builder.central_beamstop_diameter_m
        )
        self._defocus_distance_view_controller = LengthParameterViewController(
            probe_builder.defocus_distance_m, default_unit=LengthUnit.MICROMETER
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


class KBMirrorViewController(ParameterViewController):
    def __init__(self, title: str, probe_builder: KBMirrorProbeBuilder) -> None:
        super().__init__()
        self._widget = GroupBoxWithPresets(title)

        for label in probe_builder.labels_for_presets():
            action = self._widget.presets_menu.addAction(label)

            if action is None:
                raise ValueError('action is None!')
            else:
                action.triggered.connect(lambda _, label=label: probe_builder.apply_presets(label))

        self._horizontal_acceptance_length_view_controller = LengthParameterViewController(
            probe_builder.horizontal_acceptance_length_m, default_unit=LengthUnit.MILLIMETER
        )
        self._horizontal_grazing_angle_view_controller = DecimalLineEditParameterViewController(
            probe_builder.horizontal_grazing_angle_rad
        )
        self._horizontal_focus_distance_view_controller = LengthParameterViewController(
            probe_builder.horizontal_focus_distance_m, default_unit=LengthUnit.MILLIMETER
        )
        self._vertical_acceptance_length_view_controller = LengthParameterViewController(
            probe_builder.vertical_acceptance_length_m, default_unit=LengthUnit.MILLIMETER
        )
        self._vertical_grazing_angle_view_controller = DecimalLineEditParameterViewController(
            probe_builder.vertical_grazing_angle_rad
        )
        self._vertical_focus_distance_view_controller = LengthParameterViewController(
            probe_builder.vertical_focus_distance_m, default_unit=LengthUnit.MILLIMETER
        )
        self._astigmatism_view_controller = LengthParameterViewController(
            probe_builder.astigmatism_m, is_signed=True, default_unit=LengthUnit.MICROMETER
        )
        self._defocus_distance_view_controller = LengthParameterViewController(
            probe_builder.defocus_distance_m, default_unit=LengthUnit.MICROMETER
        )

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow(
            'Horizontal Acceptance Length:',
            self._horizontal_acceptance_length_view_controller.get_widget(),
        )
        layout.addRow(
            'Horizontal Grazing Angle [rad]:',
            self._horizontal_grazing_angle_view_controller.get_widget(),
        )
        layout.addRow(
            'Horizontal Focus Distance:',
            self._horizontal_focus_distance_view_controller.get_widget(),
        )
        layout.addRow(
            'Vertical Acceptance Length:',
            self._vertical_acceptance_length_view_controller.get_widget(),
        )
        layout.addRow(
            'Vertical Grazing Angle [rad]:',
            self._vertical_grazing_angle_view_controller.get_widget(),
        )
        layout.addRow(
            'Vertical Focus Distance:',
            self._vertical_focus_distance_view_controller.get_widget(),
        )
        layout.addRow('Astigmatism:', self._astigmatism_view_controller.get_widget())
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
        self._diameter_view_controller = LengthParameterViewController(probe_builder.diameter_m)

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
        self._width_view_controller = LengthParameterViewController(probe_builder.width_m)
        self._height_view_controller = LengthParameterViewController(probe_builder.height_m)

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


class IncoherentModeStrategyViewController(ParameterViewController, Observer):
    """Show only the parameters the selected incoherent-mode strategy actually reads.

    The decay profile is on the pages rather than beside the selector because the
    Gaussian-Schell model predicts its own mode spectrum from the source coherence and
    never consults a decay profile, so leaving those controls up under it would offer a
    knob that does nothing.
    """

    def __init__(self, probe_builder: ProbeSequenceBuilder) -> None:
        super().__init__()
        self._strategy = probe_builder.incoherent_mode_strategy
        self._widget = QStackedWidget()
        # The view controllers own their widgets, so the pages keep them alive.
        self._view_controllers: list[ParameterViewController] = []

        decay_rows: list[tuple[str, ParameterViewController]] = [
            (
                'Decay Type:',
                DecayTypeParameterViewController(probe_builder.incoherent_mode_decay_type),
            ),
            (
                'Decay Ratio:',
                DecimalSliderParameterViewController(probe_builder.incoherent_mode_decay_ratio),
            ),
        ]
        # A widget belongs to one page only, so each page binds its own controllers to
        # the shared parameters; they observe it and stay in step.
        more_decay_rows: list[tuple[str, ParameterViewController]] = [
            (
                'Decay Type:',
                DecayTypeParameterViewController(probe_builder.incoherent_mode_decay_type),
            ),
            (
                'Decay Ratio:',
                DecimalSliderParameterViewController(probe_builder.incoherent_mode_decay_ratio),
            ),
        ]

        self._pages: dict[str, QWidget] = {
            'MomentPolynomial': self._build_page(
                [
                    *decay_rows,
                    (
                        'Damping Width:',
                        DecimalLineEditParameterViewController(
                            probe_builder.moment_polynomial_damping_width
                        ),
                    ),
                ]
            ),
            'RandomPhaseRamp': self._build_page(more_decay_rows),
            'GaussianSchell': self._build_page(
                [
                    (
                        'Beam Width:',
                        LengthParameterViewController(probe_builder.gaussian_schell_beam_width_m),
                    ),
                    (
                        'Beam Height:',
                        LengthParameterViewController(probe_builder.gaussian_schell_beam_height_m),
                    ),
                    (
                        'Coherence Width:',
                        LengthParameterViewController(
                            probe_builder.gaussian_schell_coherence_width_m
                        ),
                    ),
                    (
                        'Coherence Height:',
                        LengthParameterViewController(
                            probe_builder.gaussian_schell_coherence_height_m
                        ),
                    ),
                ]
            ),
        }

        for page in self._pages.values():
            self._widget.addWidget(page)

        self._strategy.add_observer(self)
        self._sync_strategy_page()

    def _build_page(self, rows: list[tuple[str, ParameterViewController]]) -> QWidget:
        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        for label, view_controller in rows:
            layout.addRow(label, view_controller.get_widget())
            self._view_controllers.append(view_controller)

        page = QWidget()
        page.setLayout(layout)
        return page

    def _sync_strategy_page(self) -> None:
        namecf = self._strategy.get_value().casefold()

        for name, page in self._pages.items():
            if name.casefold() == namecf:
                self._widget.setCurrentWidget(page)
                return

        # An unrecognized name builds the first strategy, so show what will be built.
        self._widget.setCurrentWidget(self._pages[INCOHERENT_MODE_STRATEGY_NAMES[0]])

    def get_widget(self) -> QWidget:
        return self._widget

    def _update(self, observable: Observable) -> None:
        if observable is self._strategy:
            self._sync_strategy_page()


class LabelViewController(ParameterViewController):
    def __init__(self, text: str) -> None:
        super().__init__()
        self._widget = QLabel(text)
        self._widget.setWordWrap(True)

    def get_widget(self) -> QWidget:
        return self._widget


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
        entropy_metrics = self._probe_item.get_entropy_metrics()
        self._table_model.set_entropy_metrics(entropy_metrics)
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
                default_unit=LengthUnit.MICROMETER,
                group=primary_mode_group,
            )
            return True
        elif isinstance(probe_builder, FresnelZonePlateProbeBuilder):
            dialog_builder.add_view_controller_to_top(
                FresnelZonePlateViewController(primary_mode_group, probe_builder)
            )
            return True
        elif isinstance(probe_builder, KBMirrorProbeBuilder):
            dialog_builder.add_view_controller_to_top(
                KBMirrorViewController(primary_mode_group, probe_builder)
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
                default_unit=LengthUnit.MICROMETER,
                group=primary_mode_group,
            )
            return True
        elif isinstance(probe_builder, SuperGaussianProbeBuilder):
            dialog_builder.add_length_widget(
                probe_builder.annular_radius_m,
                'Annular Radius:',
                default_unit=LengthUnit.MICROMETER,
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
        expand_only_tool_tip = (
            'Modes are only ever added, never removed. A probe that already has'
            ' more modes than this keeps the ones it has.'
        )

        incoherent_modes_group = 'Incoherent (Mixed State) Modes'
        dialog_builder.add_spin_box(
            probe_builder.num_incoherent_modes,
            'Number of Modes:',
            tool_tip=expand_only_tool_tip,
            group=incoherent_modes_group,
        )
        dialog_builder.add_check_box(
            probe_builder.orthogonalize_incoherent_modes,
            'Orthogonalize Modes:',
            group=incoherent_modes_group,
        )
        dialog_builder.add_combo_box(
            probe_builder.incoherent_mode_strategy,
            INCOHERENT_MODE_STRATEGY_NAMES,
            'Strategy:',
            group=incoherent_modes_group,
        )
        dialog_builder.add_view_controller(
            IncoherentModeStrategyViewController(probe_builder),
            '',
            group=incoherent_modes_group,
        )

        coherent_modes_group = 'Coherent (OPR) Modes'
        dialog_builder.add_spin_box(
            probe_builder.num_coherent_modes,
            'Number of Modes:',
            tool_tip=expand_only_tool_tip,
            group=coherent_modes_group,
        )
        dialog_builder.add_combo_box(
            probe_builder.opr_weight_policy,
            [policy.name for policy in OPRWeightPolicy],
            'Weight Policy:',
            tool_tip=(
                'How OPR weights loaded from a file carry over when they were solved for'
                ' a different number of probe positions than this run has.'
            ),
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

        # A from-file probe is conditioned on the way in, so the mode parameters
        # do something and belong in the dialog. A from-memory probe holds an
        # already-conditioned probe -- reconstruction output, or a product loaded
        # from file -- whose mode parameters are deliberately inert, so offering
        # controls that would silently do nothing is worse than offering none.
        has_primary_mode = self._append_primary_mode(probe_builder, dialog_builder)

        if has_primary_mode or isinstance(probe_builder, FromFileProbeBuilder):
            self._append_additional_modes(probe_builder, dialog_builder)
        else:
            dialog_builder.add_view_controller_to_bottom(
                LabelViewController(f'"{builder_name}" has no editable parameters.')
            )

        return dialog_builder.build_dialog(title, parent)
