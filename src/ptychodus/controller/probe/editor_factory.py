from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe import OPRWeightPolicy
from ptychodus.api.parameters import BooleanParameter, StringParameter

from ...model.product.probe import (
    INCOHERENT_MODE_STRATEGY_NAMES,
    FromFileProbeBuilder,
    FromMemoryProbeBuilder,
    PluginProbeBuilder,
    ProbeModeDecayType,
    ProbeRepositoryItem,
    ProbeEnsembleBuilder,
)
from ...view.widgets import GroupBoxWithPresets
from ..make_editable import MakeEditableViewController
from ..parameters import (
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LengthParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
    create_parameter_view_controller,
)
from .metrics import ProbeMetricsTableModel

__all__ = [
    'ProbeEditorViewControllerFactory',
]


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

    def __init__(self, probe_builder: ProbeEnsembleBuilder) -> None:
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

        derivative_decay_rows: list[tuple[str, ParameterViewController]] = [
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
            'Derivative': self._build_page(derivative_decay_rows),
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


class EstimateSizeViewController(ParameterViewController):
    """A button that sets a parametric probe's size from the measured patterns."""

    def __init__(self, item: ProbeRepositoryItem) -> None:
        super().__init__()
        self._item = item
        self._widget = QPushButton('Estimate from Diffraction')
        self._widget.setToolTip(
            'Set the size from the second moments of the mean diffraction pattern, at the '
            'defocus distance above. Needs far-field data.'
        )
        self._widget.setEnabled(item.can_estimate_size_from_diffraction())
        self._widget.clicked.connect(self._estimate)

    def get_widget(self) -> QWidget:
        return self._widget

    def _estimate(self) -> None:
        self._item.estimate_size_from_diffraction()


class DeclaredParametersViewController(ParameterViewController):
    """The rows a generator declares, in a group offering whichever presets fit it."""

    def __init__(self, title: str, item: ProbeRepositoryItem, builder: PluginProbeBuilder) -> None:
        super().__init__()
        self._view_controllers: list[ParameterViewController] = []
        layout = QFormLayout()

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

        if builder.estimates_size:
            estimate_view_controller = EstimateSizeViewController(item)
            self._view_controllers.append(estimate_view_controller)
            layout.addRow('', estimate_view_controller.get_widget())

        preset_labels = list(builder.labels_for_presets())

        if preset_labels:
            group_box = GroupBoxWithPresets(title)

            for label in preset_labels:
                action = group_box.presets_menu.addAction(label)

                if action is None:
                    raise ValueError('action is None!')

                action.triggered.connect(lambda _, label=label: builder.apply_presets(label))

            layout.setContentsMargins(0, 0, 0, 0)
            group_box.contents.setLayout(layout)
            self._widget: QWidget = group_box
        else:
            plain_group_box = QGroupBox(title)
            plain_group_box.setLayout(layout)
            self._widget = plain_group_box

    def get_widget(self) -> QWidget:
        return self._widget


class ProbeEditorViewControllerFactory:
    def _append_primary_mode(
        self,
        item: ProbeRepositoryItem,
        dialog_builder: ParameterViewBuilder,
    ) -> bool:
        """Add the rows the item's generator declares, returning whether it has any.

        One branch for every probe model, because a generator describes its own
        parameters: a model added later needs no branch of its own. The from-file and
        from-memory builders describe nothing.
        """
        probe_builder = item.get_builder()

        if isinstance(probe_builder, PluginProbeBuilder):
            dialog_builder.add_view_controller_to_top(
                DeclaredParametersViewController('Primary Mode', item, probe_builder)
            )
            return True

        return False

    def _append_additional_modes(
        self,
        probe_builder: ProbeEnsembleBuilder,
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
        # from file -- whose mode parameters are deliberately inert until it is made
        # editable, so offering controls that would silently do nothing is worse than
        # offering the button that makes them apply.
        has_primary_mode = self._append_primary_mode(item, dialog_builder)
        is_editable_result = (
            isinstance(probe_builder, FromMemoryProbeBuilder) and probe_builder.is_editable
        )

        if (
            has_primary_mode
            or is_editable_result
            or isinstance(probe_builder, FromFileProbeBuilder)
        ):
            self._append_additional_modes(probe_builder, dialog_builder)
        elif item.can_make_editable():
            dialog_builder.add_view_controller_to_bottom(
                MakeEditableViewController(
                    item.make_editable,
                    lambda: self.create_editor_dialog(item_name, item, parent).open(),
                    'Let this result take on more incoherent or coherent modes. The '
                    'counts start at the modes it has, so nothing changes until they are '
                    'raised.',
                )
            )
        else:
            dialog_builder.add_view_controller_to_bottom(
                LabelViewController(f'"{builder_name}" has no editable parameters.')
            )

        return dialog_builder.build_dialog(title, parent)
