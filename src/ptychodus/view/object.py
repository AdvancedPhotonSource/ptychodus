from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .image import ImageView


def _box_widget(title: str, widget: QWidget) -> QWidget:
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)

    box = QGroupBox(title)
    box.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    box.setLayout(layout)

    return box


class FourierRingCorrelationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product1_label = QLabel('Product 1:')
        self.product1_combo_box = QComboBox()
        self.product2_label = QLabel('Product 2:')
        self.product2_combo_box = QComboBox()
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.frc_axes, self.ssnr_axes = self.figure.subplots(2, 1, sharex=True)

        self.auc_label = QLabel('—')
        self.average_ssnr_label = QLabel('—')
        self.resolution_label = QLabel('—')

        inputs_layout = QGridLayout()
        inputs_layout.addWidget(self.product1_label, 0, 0)
        inputs_layout.addWidget(self.product1_combo_box, 0, 1)
        inputs_layout.addWidget(self.product2_label, 1, 0)
        inputs_layout.addWidget(self.product2_combo_box, 1, 1)
        inputs_layout.setColumnStretch(1, 1)
        self.inputs_group = QGroupBox('Inputs')
        self.inputs_group.setLayout(inputs_layout)

        metrics_layout = QFormLayout()
        metrics_layout.addRow('Area under FRC curve:', self.auc_label)
        metrics_layout.addRow('Average SSNR:', self.average_ssnr_label)
        metrics_layout.addRow('Resolution @ SNR=2:', self.resolution_label)
        self.metrics_group = QGroupBox('Metrics')
        self.metrics_group.setLayout(metrics_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.inputs_group, stretch=1)
        bottom_layout.addWidget(self.metrics_group)

        layout = QVBoxLayout()
        layout.addWidget(self.navigation_toolbar)
        layout.addWidget(self.figure_canvas)
        layout.addLayout(bottom_layout)
        self.setLayout(layout)


class FourierAnalysisDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.real_space_view = ImageView(add_fourier_tool=True)
        self.reciprocal_space_view = ImageView()
        self.status_bar = QStatusBar()

        real_space_box = _box_widget('Real Space', self.real_space_view)
        reciprocal_space_box = _box_widget('Reciprocal Space', self.reciprocal_space_view)

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
        self.save_button = QPushButton('Save')

        layout = QFormLayout()
        layout.addRow('Left Circular:', self.lcirc_combo_box)
        layout.addRow('Right Circular:', self.rcirc_combo_box)
        layout.addRow(self.save_button)
        self.setLayout(layout)


class XMCDDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.structural_view = ImageView()
        self.magnetic_view = ImageView()
        self.parameters_view = XMCDParametersView()
        self.status_bar = QStatusBar()

        structural_box = _box_widget('Structural', self.structural_view)
        magnetic_box = _box_widget('Magnetic', self.magnetic_view)

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(structural_box, stretch=1)
        contents_layout.addWidget(magnetic_box, stretch=1)
        contents_layout.addWidget(self.parameters_view)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
