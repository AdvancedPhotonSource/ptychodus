from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .dashboard import DashboardDialog, MetricsGroupView, MetricsTableView, PlotView
from .image import ImageView, box_image_view
from .visualization import VisualizationParametersView, VisualizationWidget


class ObjectMetricsDialog(DashboardDialog):
    """What one object looks like over the region its probe positions illuminated."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product_combo_box = QComboBox()
        self.layer_combo_box = QComboBox()
        self.inputs_layout.addRow('Product:', self.product_combo_box)
        self.inputs_layout.addRow('Layer:', self.layer_combo_box)

        self.ambiguities_group = MetricsGroupView('Phase Ambiguities')
        self.ambiguities_group.setToolTip('Fitted to the projection and applied to layer 0')
        self.phase_offset_label = self.ambiguities_group.add_row('Phase Offset [rad]')
        self.phase_ramp_x_label = self.ambiguities_group.add_row('Phase Ramp X [rad/µm]')
        self.phase_ramp_y_label = self.ambiguities_group.add_row('Phase Ramp Y [rad/µm]')

        self.region_group = MetricsGroupView('Scoring Region')
        self.region_size_label = self.region_group.add_row('Size [px]')
        self.region_fraction_label = self.region_group.add_row(
            'Fraction of Object [%]', 'Share of the object array the illuminated region covers'
        )

        self.statistics_group = MetricsGroupView('Statistics')
        self.amplitude_mean_label = self.statistics_group.add_row('Amplitude Mean')
        self.amplitude_std_label = self.statistics_group.add_row('Amplitude Std Dev')
        self.amplitude_minimum_label = self.statistics_group.add_row('Amplitude Minimum')
        self.amplitude_maximum_label = self.statistics_group.add_row('Amplitude Maximum')
        self.phase_mean_label = self.statistics_group.add_row(
            'Phase Mean [rad]', 'Intensity-weighted circular mean'
        )
        self.phase_deviation_label = self.statistics_group.add_row(
            'Phase Std Dev [rad]', 'Intensity-weighted circular standard deviation'
        )
        self.phase_peak_to_valley_label = self.statistics_group.add_row(
            'Phase Peak-to-Valley [rad]'
        )

        self.layers_group = QGroupBox('Layers')
        self.layers_table_view = MetricsTableView(
            ['Depth [µm]', 'Phase Std Dev [rad]', 'Amplitude Mean']
        )
        layers_layout = QVBoxLayout()
        layers_layout.addWidget(self.layers_table_view)
        self.layers_group.setLayout(layers_layout)

        for group in (self.ambiguities_group, self.region_group, self.statistics_group):
            self.numbers_layout.addWidget(group)

        self.numbers_layout.addWidget(self.layers_group)
        self.finish_numbers()

        self.amplitude_widget = VisualizationWidget('Amplitude')
        self.phase_widget = VisualizationWidget('Phase')
        self.images_layout.addWidget(self.amplitude_widget, 0, 0)
        self.images_layout.addWidget(self.phase_widget, 0, 1)

        self.plot_view = PlotView()
        self.plot_view.figure.clear()
        self.psd_axes, self.amplitude_histogram_axes, self.phase_histogram_axes = (
            self.plot_view.figure.subplots(1, 3)
        )
        self.plots_layout.addWidget(self.plot_view)


class ObjectCompareDialog(DashboardDialog):
    """How a test object agrees with a reference object."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.reference_combo_box = QComboBox()
        self.test_combo_box = QComboBox()
        self.convention_combo_box = QComboBox()
        self.inputs_layout.addRow('Reference:', self.reference_combo_box)
        self.inputs_layout.addRow('Test:', self.test_combo_box)
        self.inputs_layout.addRow('Pair:', self.convention_combo_box)
        self.inputs_layout.addRow(QLabel('Multislice objects are compared by their projections.'))

        self.resolution_group = MetricsGroupView('Resolution')
        self.resolution_at_snr_label = self.resolution_group.add_row('At SNR = 2 [nm]')
        self.resolution_at_half_bit_label = self.resolution_group.add_row('At ½-Bit [nm]')
        self.area_under_curve_label = self.resolution_group.add_row('Area Under FRC')
        self.average_snr_label = self.resolution_group.add_row('Average SSNR')

        self.registration_group = MetricsGroupView('Registration')
        self.shift_x_label = self.registration_group.add_row('Shift X [px]')
        self.shift_y_label = self.registration_group.add_row('Shift Y [px]')
        self.scale_label = self.registration_group.add_row('Scale')
        self.phase_offset_label = self.registration_group.add_row('Phase Offset [rad]')
        self.phase_ramp_x_label = self.registration_group.add_row('Phase Ramp X [rad/µm]')
        self.phase_ramp_y_label = self.registration_group.add_row('Phase Ramp Y [rad/µm]')
        self.resampled_label = self.registration_group.add_row('Test Resampled')

        self.pixelwise_group = QGroupBox('Pixelwise Agreement')
        self.pixelwise_table_view = MetricsTableView(['Amplitude', 'Phase'])
        pixelwise_layout = QVBoxLayout()
        pixelwise_layout.addWidget(self.pixelwise_table_view)
        self.pixelwise_group.setLayout(pixelwise_layout)

        self.visualization_parameters_view = VisualizationParametersView()

        self.numbers_layout.addWidget(self.resolution_group)
        self.numbers_layout.addWidget(self.registration_group)
        self.numbers_layout.addWidget(self.pixelwise_group)
        self.numbers_layout.addWidget(self.visualization_parameters_view)
        self.finish_numbers()

        self.reference_widget = VisualizationWidget('Reference')
        self.test_widget = VisualizationWidget('Test, Registered')
        self.difference_widget = VisualizationWidget('Difference')
        self.images_layout.addWidget(self.reference_widget, 0, 0)
        self.images_layout.addWidget(self.test_widget, 0, 1)
        self.images_layout.addWidget(self.difference_widget, 0, 2)

        self.plot_view = PlotView()
        self.plot_view.figure.clear()
        self.frc_axes, self.ssnr_axes, self.psd_axes = self.plot_view.figure.subplots(
            1, 3, sharex=True
        )
        self.plots_layout.addWidget(self.plot_view)


class FourierAnalysisDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.real_space_view = ImageView(add_fourier_tool=True)
        self.reciprocal_space_view = ImageView()
        self.status_bar = QStatusBar()

        real_space_box = box_image_view('Real Space', self.real_space_view)
        reciprocal_space_box = box_image_view('Reciprocal Space', self.reciprocal_space_view)

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(real_space_box)
        contents_layout.addWidget(reciprocal_space_box)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)


class XMCDParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)

        self.lcirc_combo_box = QComboBox()
        self.rcirc_combo_box = QComboBox()
        self.add_button = QPushButton('Add to Products')

        layout = QFormLayout()
        layout.addRow('Left Circular:', self.lcirc_combo_box)
        layout.addRow('Right Circular:', self.rcirc_combo_box)
        layout.addRow(self.add_button)
        self.setLayout(layout)


class XMCDDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.structural_view = ImageView()
        self.magnetic_view = ImageView()
        self.parameters_view = XMCDParametersView()
        self.status_bar = QStatusBar()

        structural_box = box_image_view('Structural', self.structural_view)
        magnetic_box = box_image_view('Magnetic', self.magnetic_view)

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(structural_box, stretch=1)
        contents_layout.addWidget(magnetic_box, stretch=1)
        contents_layout.addWidget(self.parameters_view)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
