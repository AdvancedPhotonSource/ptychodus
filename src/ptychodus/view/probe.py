from collections.abc import Mapping

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ptychodus.api.probe import OPRWeightPolicy

from .image import ImageView, box_image_view
from .visualization import VisualizationParametersView, VisualizationWidget
from .widgets import DecimalLineEdit


_OPR_WEIGHT_POLICY_LABELS: Mapping[OPRWeightPolicy, tuple[str, str]] = {
    OPRWeightPolicy.KEEP: (
        'Use as is',
        'Keep the weights unchanged. Available only when the counts match.',
    ),
    OPRWeightPolicy.AVERAGE: (
        'Average the weights',
        'Give every probe position the mean of the loaded weights, keeping the modes.',
    ),
    OPRWeightPolicy.REINITIALIZE: (
        'Reinitialize the weights',
        'Keep the modes, but start the weights where a fresh OPR run would.',
    ),
    OPRWeightPolicy.COLLAPSE: (
        'Remove OPR, keep the average probe',
        'Combine the modes under the mean weights into one, then drop the OPR basis.',
    ),
    OPRWeightPolicy.DISCARD: (
        'Remove OPR, keep the primary mode',
        'Keep the dominant coherent mode and drop the rest along with the weights.',
    ),
}


class OPRWeightPolicyDialog(QDialog):
    """Asks how a loaded probe's OPR weights should carry over to this run."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Probe Has OPR Modes')
        self.setModal(True)

        self._counts_label = QLabel()
        self._counts_label.setWordWrap(True)
        self._button_group = QButtonGroup()
        self._buttons: dict[int, QRadioButton] = {}
        self.button_box = QDialogButtonBox()

        layout = QVBoxLayout()
        layout.addWidget(self._counts_label)

        for policy, (label, description) in _OPR_WEIGHT_POLICY_LABELS.items():
            button = QRadioButton(label)
            button.setToolTip(description)
            self._button_group.addButton(button, policy.value)
            self._buttons[policy.value] = button
            layout.addWidget(button)

            caption = QLabel(description)
            caption.setWordWrap(True)
            caption.setIndent(20)
            caption.setEnabled(False)
            layout.addWidget(caption)

        self._button_group.setExclusive(True)

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

    def set_counts(
        self, num_weight_rows: int, num_scan_points: int, num_coherent_modes: int
    ) -> None:
        """State what the file holds and what this run needs, and gate "use as is" on it."""
        counts_agree = num_weight_rows == num_scan_points
        agreement = 'matches' if counts_agree else 'does not match'
        self._counts_label.setText(
            f'This probe has {num_coherent_modes} coherent (OPR) mode(s) with weights for'
            f' {num_weight_rows} probe position(s), which {agreement} the'
            f' {num_scan_points} probe position(s) in this run.'
        )
        self._buttons[OPRWeightPolicy.KEEP.value].setEnabled(counts_agree)

    def set_policy(self, policy: OPRWeightPolicy) -> None:
        button = self._buttons[policy.value]

        if button.isEnabled():
            button.setChecked(True)
        else:
            self._buttons[OPRWeightPolicy.AVERAGE.value].setChecked(True)

    def get_policy(self) -> OPRWeightPolicy:
        return OPRWeightPolicy(self._button_group.checkedId())


class OPRModeStatisticsView(QGroupBox):
    """Per-mode weight statistics, and the two numbers that summarize them.

    The table says how each coherent mode behaves across the scan; the two summary rows
    beneath it answer whether the OPR basis is earning its degrees of freedom at all,
    and how many of its modes are doing the work.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Mode Statistics', parent)
        self.relative_variation_label = QLabel('—')
        self.effective_mode_count_label = QLabel('—')

        self.relative_variation_label.setToolTip(
            'Share of the composed mode’s power that varies across the scan.'
            ' Near zero means the OPR basis is not modeling anything.'
        )
        self.effective_mode_count_label.setToolTip(
            'Coherent modes needed to account for most of that variation.'
            ' Fewer than the basis holds means the extra modes are idle.'
        )

        self._table_layout = QGridLayout()
        self._table_layout.addWidget(QLabel('Mode'), 0, 0)
        self._table_layout.addWidget(QLabel('Mean'), 0, 1, Qt.AlignmentFlag.AlignCenter)
        self._table_layout.addWidget(QLabel('Std Dev'), 0, 2, Qt.AlignmentFlag.AlignCenter)
        self._table_layout.addWidget(QLabel('Variance [%]'), 0, 3, Qt.AlignmentFlag.AlignCenter)
        self._table_layout.setColumnStretch(1, 1)
        self._table_layout.setColumnStretch(2, 1)
        self._table_layout.setColumnStretch(3, 1)
        self._rows: list[tuple[QLabel, QLabel, QLabel, QLabel]] = []

        summary_layout = QFormLayout()
        summary_layout.addRow('Varying Power [%]:', self.relative_variation_label)
        summary_layout.addRow('Modes Needed:', self.effective_mode_count_label)

        layout = QVBoxLayout()
        layout.addLayout(self._table_layout)
        layout.addLayout(summary_layout)
        self.setLayout(layout)

    def set_num_modes(self, num_modes: int) -> None:
        """Grow or shrink the table to ``num_modes`` rows.

        Rows are built once and reused, since the mode count only changes when a
        different probe is analyzed.
        """
        while len(self._rows) < num_modes:
            # One-based, matching how the probe tree names modes.
            row = len(self._rows) + 1
            labels = (QLabel(f'{row}'), QLabel('—'), QLabel('—'), QLabel('—'))

            for column, label in enumerate(labels):
                if column > 0:
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                self._table_layout.addWidget(label, row, column)

            self._rows.append(labels)

        for index, labels in enumerate(self._rows):
            for label in labels:
                label.setVisible(index < num_modes)

    def set_mode(self, mode: int, mean: str, deviation: str, variance: str) -> None:
        _, mean_label, deviation_label, variance_label = self._rows[mode]
        mean_label.setText(mean)
        deviation_label.setText(deviation)
        variance_label.setText(variance)

    def clear_modes(self) -> None:
        for _, mean_label, deviation_label, variance_label in self._rows:
            mean_label.setText('—')
            deviation_label.setText('—')
            variance_label.setText('—')

        self.relative_variation_label.setText('—')
        self.effective_mode_count_label.setText('—')


class OPRModeDisplayView(QGroupBox):
    """Chooses what the image pane renders at the selected probe position.

    The composed mode barely changes from position to position -- OPR variation is
    typically a fraction of a percent -- so on a fixed color scale it looks frozen.
    Subtracting the across-scan mean leaves only what the weights move, which is the
    view in which the variation is actually visible. The basis modes themselves are
    position-independent and say what *kind* of variation each one models.
    """

    def __init__(self, mode_spin_box: QWidget, parent: QWidget | None = None) -> None:
        super().__init__('Display', parent)
        self.composed_button = QRadioButton('Composed Probe')
        self.deviation_button = QRadioButton('Deviation from Mean')
        self.basis_button = QRadioButton('Basis Mode')

        self.composed_button.setToolTip(
            'Incoherent mode 0 after the coherent weighted sum, at the selected position'
        )
        self.deviation_button.setToolTip(
            'That mode minus its across-scan mean: what the OPR weights actually change'
        )
        self.basis_button.setToolTip(
            'One coherent basis mode on its own; the same at every probe position'
        )

        layout = QFormLayout()
        layout.addRow(self.composed_button)
        layout.addRow(self.deviation_button)
        layout.addRow(self.basis_button)
        layout.addRow('Coherent Mode:', mode_spin_box)
        self.setLayout(layout)


class OPRModeCurvesView(QGroupBox):
    """Picks which weight curves are plotted and what colors the scan grid."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Curves', parent)
        self.color_by_combo_box = QComboBox()
        self.normalize_check_box = QCheckBox('Normalize Weight Curves')
        self.mode_list_widget = QListWidget()

        self.color_by_combo_box.setToolTip('Quantity the scan grid colors each position by')
        self.normalize_check_box.setToolTip(
            'Rescale each weight curve to unit standard deviation so a weak mode’s'
            ' shape can be compared against a strong one'
        )
        self.mode_list_widget.setToolTip('Coherent modes whose weights are plotted')
        self.mode_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        layout = QFormLayout()
        layout.addRow('Color By:', self.color_by_combo_box)
        layout.addRow(self.normalize_check_box)
        layout.addRow(self.mode_list_widget)
        self.setLayout(layout)


class OPRModePlotView(QWidget):
    """Toolbar over a single matplotlib canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.navigation_toolbar)
        layout.addWidget(self.figure_canvas)
        self.setLayout(layout)


class OPRModeDialog(QDialog):
    """How a probe's coherent (OPR) basis varies over a scan.

    The image pane and the two plots are three views of one selection: the slider picks
    a probe position, and the series plot and scan grid mark where it sits.
    """

    def __init__(
        self,
        mode_spin_box: QWidget,
        play_button: QWidget,
        frame_rate_spin_box: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.probe_view = ImageView()
        self.display_view = OPRModeDisplayView(mode_spin_box)
        self.series_plot_view = OPRModePlotView()
        self.scan_grid_plot_view = OPRModePlotView()
        self.statistics_view = OPRModeStatisticsView()
        self.curves_view = OPRModeCurvesView()
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_label = QLabel()
        self.save_button = QPushButton('Save')
        self.status_bar = QStatusBar()

        self.position_slider.setToolTip('Probe Position')
        self.save_button.setToolTip('Save the per-position series to a NumPy archive')

        # The reading gains digits as the scan index grows; reserve room for the widest
        # one so the control strip does not shift while playing.
        self.position_label.setMinimumWidth(
            self.position_label.fontMetrics().horizontalAdvance('Position 000000 / 000000')
        )
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # `ImageView` hangs its ribbon off `setMenuBar`, which keeps the ribbon out of the
        # layout's width calculation. The probe pane sits in the narrow column here, so it
        # has to carry the ribbon's width itself or the Data Range group is clipped.
        self.probe_view.setMinimumWidth(self.probe_view.image_ribbon.minimumSizeHint().width())

        probe_widget = QWidget()
        probe_layout = QVBoxLayout()
        probe_layout.setContentsMargins(0, 0, 0, 0)
        probe_layout.addWidget(box_image_view('Probe', self.probe_view), 1)
        probe_layout.addWidget(self.display_view)
        probe_widget.setLayout(probe_layout)

        series_group = QGroupBox('Across the Scan')
        series_layout = QVBoxLayout()
        series_layout.addWidget(self.series_plot_view)
        series_group.setLayout(series_layout)

        # The image-to-plot ratio is a matter of what the user is looking for, so it is
        # theirs to set rather than fixed here.
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.addWidget(probe_widget)
        left_splitter.addWidget(series_group)
        left_splitter.setStretchFactor(0, 2)
        left_splitter.setStretchFactor(1, 1)

        scan_grid_group = QGroupBox('Across the Scan Grid')
        scan_grid_layout = QVBoxLayout()
        scan_grid_layout.addWidget(self.scan_grid_plot_view)
        scan_grid_group.setLayout(scan_grid_layout)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(scan_grid_group, 1)
        right_layout.addWidget(self.statistics_view)
        right_layout.addWidget(self.curves_view)

        position_layout = QHBoxLayout()
        position_layout.setContentsMargins(0, 0, 0, 0)
        position_layout.addWidget(self.position_slider, 1)
        position_layout.addWidget(play_button)
        position_layout.addWidget(frame_rate_spin_box)
        position_layout.addWidget(self.position_label)
        position_layout.addWidget(self.save_button)

        contents_layout = QGridLayout()
        contents_layout.addWidget(left_splitter, 0, 0)
        contents_layout.addLayout(right_layout, 0, 1)
        contents_layout.addLayout(position_layout, 1, 0, 1, 2)
        contents_layout.setColumnStretch(0, 2)
        contents_layout.setColumnStretch(1, 1)
        contents_layout.setRowStretch(0, 1)
        contents_layout.setRowStretch(1, 0)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)


class ProbeMetricsView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('XY Probe Metrics', parent)
        self.major_axis_tilt_label = QLabel('N/A')
        self.minor_axis_tilt_label = QLabel('N/A')
        self.fwhm_major_axis_label = QLabel('N/A')
        self.fwhm_minor_axis_label = QLabel('N/A')
        self.rms_major_axis_label = QLabel('N/A')
        self.rms_minor_axis_label = QLabel('N/A')
        self.encircled_energy_diameter_label = QLabel('N/A')

        layout = QGridLayout()
        layout.addWidget(QLabel('Major Axis'), 0, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel('Minor Axis'), 0, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel('Tilt [deg]:'), 1, 0)
        layout.addWidget(self.major_axis_tilt_label, 1, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.minor_axis_tilt_label, 1, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel('FWHM [nm]:'), 2, 0)
        layout.addWidget(self.fwhm_major_axis_label, 2, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fwhm_minor_axis_label, 2, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel('RMS [nm]:'), 3, 0)
        layout.addWidget(self.rms_major_axis_label, 3, 1, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.rms_minor_axis_label, 3, 2, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(QLabel('Encircled Energy Diameter [nm]:'), 4, 0)
        layout.addWidget(
            self.encircled_energy_diameter_label, 4, 1, 1, 2, Qt.AlignmentFlag.AlignCenter
        )
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.setLayout(layout)


class ProbeVisualizationView(QGroupBox):
    """Chooses what the propagation dialog renders: the mode-summed intensity, or one
    incoherent mode's complex wavefield.

    Complex is the only branch that can select a mode -- summing mutually incoherent
    modes is meaningful in intensity alone -- so the spin box follows the radio.
    """

    def __init__(self, mode_spin_box: QWidget, parent: QWidget | None = None) -> None:
        super().__init__('Probe', parent)
        self.intensity_button = QRadioButton('Intensity')
        self.complex_button = QRadioButton('Complex')

        self.intensity_button.setToolTip('Incoherent sum over all probe modes')
        self.complex_button.setToolTip('Complex wavefield of a single probe mode')

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.intensity_button)
        button_layout.addWidget(self.complex_button)

        layout = QFormLayout()
        layout.addRow('Incoherent Mode:', mode_spin_box)
        layout.addRow(button_layout)
        self.setLayout(layout)


class ProbePropagationDialog(QDialog):
    def __init__(
        self,
        begin_coordinate_widget: QWidget,
        end_coordinate_widget: QWidget,
        num_steps_spin_box: QWidget,
        mode_spin_box: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.xy_view = ImageView()
        self.metrics_view = ProbeMetricsView()
        self.probe_view = ProbeVisualizationView(mode_spin_box)
        self.zx_view = VisualizationWidget('ZX Plane')
        self.zy_view = VisualizationWidget('ZY Plane')
        self.propagate_button = QPushButton('Propagate')
        self.focus_button = QPushButton('Focus')
        self.save_button = QPushButton('Save')
        self.coordinate_slider = QSlider(Qt.Orientation.Horizontal)
        self.coordinate_label = QLabel()
        self.status_bar = QStatusBar()

        self.coordinate_slider.setToolTip('Propagation Plane')

        # The label text changes width as the coordinate crosses unit boundaries; reserve
        # room for the widest reading so the control strip does not shift while dragging.
        self.coordinate_label.setMinimumWidth(
            self.coordinate_label.fontMetrics().horizontalAdvance('-000.000 mm')
        )
        self.coordinate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # `ImageView` hangs its ribbon off `setMenuBar`, which keeps the ribbon out of the
        # layout's width calculation. In this dialog the XY pane sits in the narrow column,
        # so it has to carry the ribbon's width itself or the Data Range group is clipped.
        self.xy_view.setMinimumWidth(self.xy_view.image_ribbon.minimumSizeHint().width())

        xy_layout = QVBoxLayout()
        xy_layout.setContentsMargins(0, 0, 0, 0)
        xy_layout.addWidget(box_image_view('XY Plane', self.xy_view), 1)
        xy_layout.addWidget(self.metrics_view)
        xy_layout.addWidget(self.probe_view)

        coordinate_layout = QHBoxLayout()
        coordinate_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_layout.addWidget(begin_coordinate_widget)
        coordinate_layout.addWidget(self.coordinate_slider, 1)
        coordinate_layout.addWidget(self.coordinate_label)
        coordinate_layout.addWidget(end_coordinate_widget)
        coordinate_layout.addWidget(num_steps_spin_box)
        coordinate_layout.addWidget(self.propagate_button)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.addWidget(self.focus_button)
        action_layout.addWidget(self.save_button)

        contents_layout = QGridLayout()
        contents_layout.addLayout(xy_layout, 0, 0, 2, 1)
        contents_layout.addWidget(self.zx_view, 0, 1)
        contents_layout.addWidget(self.zy_view, 1, 1)
        contents_layout.addLayout(action_layout, 2, 0)
        contents_layout.addLayout(coordinate_layout, 2, 1)
        contents_layout.setColumnStretch(0, 1)
        contents_layout.setColumnStretch(1, 2)
        contents_layout.setRowStretch(0, 1)
        contents_layout.setRowStretch(1, 1)
        contents_layout.setRowStretch(2, 0)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)


class ProbeFocusDialog(QDialog):
    """Metric-versus-z curves for a propagated probe, with the focal plane each metric
    implies.

    The table on the left is both the legend and the readout: checking a row plots that
    curve, and selecting a row makes it the one the focus marker and **Go To Focus**
    act on.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metric_table_view = QTableView()
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.axes = self.figure.add_subplot(111)
        self.normalize_check_box = QCheckBox('Normalize Curves')
        self.focus_label = QLabel()
        self.go_to_focus_button = QPushButton('Go To Focus')

        self.normalize_check_box.setChecked(True)
        self.normalize_check_box.setToolTip(
            'Rescale each curve to [0, 1] so metrics in different units can be compared'
        )
        self.go_to_focus_button.setToolTip(
            'Move the propagation plane to the focus of the selected metric'
        )

        # Rows are picked whole: the current row selects the primary metric, and
        # individual cells are never edited -- the check box in column 0 is the only
        # interactive part. Per-column resize modes are left to the controller, which
        # sets them once a model exists; addressing a section before then is a crash.
        self.metric_table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.metric_table_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.metric_table_view.verticalHeader().hide()

        focus_layout = QFormLayout()
        focus_layout.addRow('Selected Metric:', self.focus_label)
        focus_layout.addRow(self.go_to_focus_button)
        focus_group = QGroupBox('Focus')
        focus_group.setLayout(focus_layout)

        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(self.metric_table_view)
        metrics_group = QGroupBox('Metrics')
        metrics_group.setLayout(metrics_layout)

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(metrics_group, 1)
        left_layout.addWidget(focus_group)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.navigation_toolbar)
        right_layout.addWidget(self.figure_canvas, 1)
        right_layout.addWidget(self.normalize_check_box)

        layout = QHBoxLayout()
        layout.addLayout(left_layout)
        layout.addLayout(right_layout, 1)
        self.setLayout(layout)


class IlluminationParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)
        self.photon_flux_line_edit = DecimalLineEdit.create_instance()
        self.exposure_time_line_edit = DecimalLineEdit.create_instance()
        self.mass_attenuation_label = QLabel('Mass Attenuation [m\u00b2/kg]:')
        self.mass_attenuation_line_edit = DecimalLineEdit.create_instance()

        layout = QFormLayout()
        layout.addRow('Photon Flux [ph/s]:', self.photon_flux_line_edit)
        layout.addRow('Exposure Time [s]:', self.exposure_time_line_edit)
        layout.addRow(self.mass_attenuation_label)
        layout.addRow(self.mass_attenuation_line_edit)
        self.setLayout(layout)


class IlluminationQuantityView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Quantity', parent)
        self.photon_number_button = QRadioButton('Photon Number')
        self.photon_fluence_button = QRadioButton('Photon Fluence [1/m\u00b2]')
        self.photon_fluence_rate_button = QRadioButton('Photon Fluence Rate [1/(s m\u00b2)]')
        self.energy_fluence_button = QRadioButton('Energy Fluence [J/m\u00b2]')
        self.energy_fluence_rate_button = QRadioButton('Energy Fluence Rate [W/m\u00b2]')
        self.dose_button = QRadioButton('Dose [Gy]')
        self.dose_rate_button = QRadioButton('Dose Rate [Gy/s]')

        layout = QVBoxLayout()
        layout.addWidget(self.photon_number_button)
        layout.addWidget(self.photon_fluence_button)
        layout.addWidget(self.photon_fluence_rate_button)
        layout.addWidget(self.energy_fluence_button)
        layout.addWidget(self.energy_fluence_rate_button)
        layout.addWidget(self.dose_button)
        layout.addWidget(self.dose_rate_button)
        self.setLayout(layout)


class IlluminationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.visualization_widget = VisualizationWidget('Visualization')
        self.parameters_view = IlluminationParametersView()
        self.quantity_view = IlluminationQuantityView()
        self.visualization_parameters_view = VisualizationParametersView()
        self.save_button = QPushButton('Save')
        self.status_bar = QStatusBar()

        parameter_layout = QVBoxLayout()
        parameter_layout.addWidget(self.parameters_view)
        parameter_layout.addWidget(self.quantity_view)
        parameter_layout.addWidget(self.visualization_parameters_view)
        parameter_layout.addWidget(self.save_button)
        parameter_layout.addStretch()

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(self.visualization_widget, 1)
        contents_layout.addLayout(parameter_layout)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)


class ProbeOverlapMetricsView(QGroupBox):
    """How densely the scan's probe footprints tile the illuminated region.

    Redundancy and the equivalent linear overlap answer "is this scan dense enough";
    the pairwise rows answer "is any single position weakly constrained", which the
    global figures average away.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Metrics', parent)
        self.areal_redundancy_label = QLabel('—')
        self.equivalent_linear_overlap_label = QLabel('—')
        self.effective_probe_diameter_label = QLabel('—')
        self.effective_step_size_label = QLabel('—')
        self.mean_pairwise_overlap_label = QLabel('—')
        self.median_pairwise_overlap_label = QLabel('—')
        self.minimum_pairwise_overlap_label = QLabel('—')
        self.maximum_pairwise_overlap_label = QLabel('—')
        self.num_positions_label = QLabel('—')

        self.minimum_pairwise_overlap_label.setToolTip(
            'A value near zero marks a scan position that no neighbor overlaps'
        )

        layout = QFormLayout()
        layout.addRow('Areal Redundancy:', self.areal_redundancy_label)
        layout.addRow('Equivalent Linear Overlap [%]:', self.equivalent_linear_overlap_label)
        layout.addRow('Effective Probe Diameter [nm]:', self.effective_probe_diameter_label)
        layout.addRow('Effective Step Size [nm]:', self.effective_step_size_label)
        layout.addRow('Pairwise Overlap Mean:', self.mean_pairwise_overlap_label)
        layout.addRow('Pairwise Overlap Median:', self.median_pairwise_overlap_label)
        layout.addRow('Pairwise Overlap Minimum:', self.minimum_pairwise_overlap_label)
        layout.addRow('Pairwise Overlap Maximum:', self.maximum_pairwise_overlap_label)
        layout.addRow('Number of Positions:', self.num_positions_label)
        self.setLayout(layout)


class ProbeOverlapDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.visualization_widget = VisualizationWidget('Redundancy')
        self.metrics_view = ProbeOverlapMetricsView()
        self.visualization_parameters_view = VisualizationParametersView()
        self.save_button = QPushButton('Save')
        self.status_bar = QStatusBar()

        parameter_layout = QVBoxLayout()
        parameter_layout.addWidget(self.metrics_view)
        parameter_layout.addWidget(self.visualization_parameters_view)
        parameter_layout.addWidget(self.save_button)
        parameter_layout.addStretch()

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(self.visualization_widget, 1)
        contents_layout.addLayout(parameter_layout)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
