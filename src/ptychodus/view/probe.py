from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .image import ImageView, box_image_view
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

        contents_layout = QGridLayout()
        contents_layout.addLayout(xy_layout, 0, 0, 2, 1)
        contents_layout.addWidget(self.zx_view, 0, 1)
        contents_layout.addWidget(self.zy_view, 1, 1)
        contents_layout.addWidget(self.save_button, 2, 0)
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
