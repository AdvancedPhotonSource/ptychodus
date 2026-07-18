from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .visualization import VisualizationParametersView, VisualizationWidget
from .widgets import DecimalLineEdit


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


class ProbePropagationParametersView(QGroupBox):
    def __init__(
        self,
        begin_coordinate_widget: QWidget,
        end_coordinate_widget: QWidget,
        num_steps_spin_box: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__('Parameters', parent)
        self.visualization_parameters_view = VisualizationParametersView()
        self.metrics_view = ProbeMetricsView()

        propagation_layout = QFormLayout()
        propagation_layout.addRow('Begin Coordinate:', begin_coordinate_widget)
        propagation_layout.addRow('End Coordinate:', end_coordinate_widget)
        propagation_layout.addRow('Number of Steps:', num_steps_spin_box)

        propagation_group_box = QGroupBox('Propagation')
        propagation_group_box.setLayout(propagation_layout)

        scroll_widget = QWidget()
        scroll_widget_layout = QVBoxLayout()
        scroll_widget_layout.setContentsMargins(0, 0, 0, 0)
        scroll_widget_layout.addWidget(self.metrics_view)
        scroll_widget_layout.addWidget(propagation_group_box)
        scroll_widget_layout.addWidget(self.visualization_parameters_view)
        scroll_widget_layout.addStretch()
        scroll_widget.setLayout(scroll_widget_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        layout = QVBoxLayout()
        layout.addWidget(scroll_area)
        self.setLayout(layout)


class ProbePropagationDialog(QDialog):
    def __init__(
        self,
        begin_coordinate_widget: QWidget,
        end_coordinate_widget: QWidget,
        num_steps_spin_box: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.xy_view = VisualizationWidget('XY Plane')
        self.zx_view = VisualizationWidget('ZX Plane')
        self.parameters_view = ProbePropagationParametersView(
            begin_coordinate_widget, end_coordinate_widget, num_steps_spin_box
        )
        self.zy_view = VisualizationWidget('ZY Plane')
        self.propagate_button = QPushButton('Propagate')
        self.save_button = QPushButton('Save')
        self.coordinate_slider = QSlider(Qt.Orientation.Horizontal)
        self.coordinate_label = QLabel()
        self.status_bar = QStatusBar()

        action_layout = QHBoxLayout()
        action_layout.addWidget(self.propagate_button)
        action_layout.addWidget(self.save_button)

        coordinate_layout = QHBoxLayout()
        coordinate_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_layout.addWidget(self.coordinate_slider)
        coordinate_layout.addWidget(self.coordinate_label)

        contents_layout = QGridLayout()
        contents_layout.addWidget(self.xy_view, 0, 0)
        contents_layout.addWidget(self.zx_view, 0, 1)
        contents_layout.addWidget(self.parameters_view, 1, 0)
        contents_layout.addWidget(self.zy_view, 1, 1)
        contents_layout.addLayout(action_layout, 2, 0)
        contents_layout.addLayout(coordinate_layout, 2, 1)
        contents_layout.setColumnStretch(0, 1)
        contents_layout.setColumnStretch(1, 2)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
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
        self.photon_fluence_rate_button = QRadioButton('Photon Fluence Rate [Hz/m\u00b2]')
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


class FluorescenceVSPIParametersView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.damping_factor_line_edit = DecimalLineEdit.create_instance()
        self.max_iterations_spin_box = QSpinBox()

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow('Damping Factor:', self.damping_factor_line_edit)
        layout.addRow('Max Iterations:', self.max_iterations_spin_box)
        self.setLayout(layout)


class FluorescencePtychozoonParametersView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.damping_factor_line_edit = DecimalLineEdit.create_instance()
        self.gradient_smoothness_line_edit = DecimalLineEdit.create_instance()
        self.max_iterations_spin_box = QSpinBox()
        self.atol_line_edit = DecimalLineEdit.create_instance()
        self.btol_line_edit = DecimalLineEdit.create_instance()
        self.checkpoint_interval_spin_box = QSpinBox()
        self.use_gpu_check_box = QCheckBox('Use GPU')
        self.gpu_device_index_spin_box = QSpinBox()

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow('Damping Factor:', self.damping_factor_line_edit)
        layout.addRow('Gradient Smoothness:', self.gradient_smoothness_line_edit)
        layout.addRow('Max Iterations:', self.max_iterations_spin_box)
        layout.addRow('A Tolerance:', self.atol_line_edit)
        layout.addRow('B Tolerance:', self.btol_line_edit)
        layout.addRow('Checkpoint Interval:', self.checkpoint_interval_spin_box)
        layout.addRow(self.use_gpu_check_box)
        layout.addRow('CUDA Device Index:', self.gpu_device_index_spin_box)
        self.setLayout(layout)


class FluorescenceTwoStepParametersView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.upscaling_strategy_combo_box = QComboBox()
        self.deconvolution_strategy_combo_box = QComboBox()

        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addRow('Upscaling Strategy:', self.upscaling_strategy_combo_box)
        layout.addRow('Deconvolution Strategy:', self.deconvolution_strategy_combo_box)
        self.setLayout(layout)


class FluorescenceParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Enhancement Strategy', parent)
        self.open_button = QPushButton('Open Measured Dataset')
        self.algorithm_combo_box = QComboBox()
        self.stacked_widget = QStackedWidget()
        self.enhance_button = QPushButton('Enhance')
        self.save_button = QPushButton('Save Enhanced Dataset')

        stacked_widget_layout = self.stacked_widget.layout()

        if stacked_widget_layout is not None:
            stacked_widget_layout.setContentsMargins(0, 0, 0, 0)

        layout = QFormLayout()
        layout.addRow(self.open_button)
        layout.addRow('Algorithm:', self.algorithm_combo_box)
        layout.addRow(self.stacked_widget)
        layout.addRow(self.enhance_button)
        layout.addRow(self.save_button)
        self.setLayout(layout)


class FluorescenceStatusView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Status', parent)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.stop_button = QPushButton('Stop')

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addLayout(progress_layout)
        self.setLayout(layout)


class FluorescenceDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.measured_widget = VisualizationWidget('Measured')
        self.enhanced_widget = VisualizationWidget('Enhanced')
        self.fluorescence_parameters_view = FluorescenceParametersView()
        self.fluorescence_channel_list_view = QListView()
        self.visualization_parameters_view = VisualizationParametersView()
        self.fluorescence_status_view = FluorescenceStatusView()
        self.status_bar = QStatusBar()

        parameter_layout = QVBoxLayout()
        parameter_layout.addWidget(self.fluorescence_parameters_view)
        parameter_layout.addWidget(self.fluorescence_channel_list_view, 1)
        parameter_layout.addWidget(self.visualization_parameters_view)
        parameter_layout.addWidget(self.fluorescence_status_view, 1)

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(self.measured_widget, 1)
        contents_layout.addWidget(self.enhanced_widget, 1)
        contents_layout.addLayout(parameter_layout)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
