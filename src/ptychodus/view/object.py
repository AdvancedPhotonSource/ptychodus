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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .image import ImageView
from .visualization import VisualizationParametersView, VisualizationWidget


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
        self.axes = self.figure.add_subplot(111)

        parameters_layout = QGridLayout()
        parameters_layout.addWidget(self.product1_label, 0, 0)
        parameters_layout.addWidget(self.product1_combo_box, 0, 1)
        parameters_layout.addWidget(self.product2_label, 0, 2)
        parameters_layout.addWidget(self.product2_combo_box, 0, 3)
        parameters_layout.setColumnStretch(1, 1)
        parameters_layout.setColumnStretch(3, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.navigation_toolbar)
        layout.addWidget(self.figure_canvas)
        layout.addLayout(parameters_layout)
        self.setLayout(layout)


class FourierAnalysisDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.real_space_view = ImageView(add_fourier_tool=True)
        # FIXME self.real_space_widget = VisualizationWidget('Real Space', add_fourier_tool=True)
        self.reciprocal_space_view = ImageView()
        # FIXME self.reciprocal_space_widget = VisualizationWidget('Reciprocal Space')
        self.status_bar = QStatusBar()

        contents_layout = QHBoxLayout()
        contents_layout.addWidget(self.real_space_view)
        contents_layout.addWidget(self.reciprocal_space_view)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)


class XMCDParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)

        self.polarization_group_box = QGroupBox('Polarization')
        self.lcirc_combo_box = QComboBox()
        self.rcirc_combo_box = QComboBox()
        self.save_button = QPushButton('Save')
        self.visualization_parameters_view = VisualizationParametersView()

        polarization_layout = QFormLayout()
        polarization_layout.addRow('Left Circular:', self.lcirc_combo_box)
        polarization_layout.addRow('Right Circular:', self.rcirc_combo_box)
        polarization_layout.addRow(self.save_button)
        self.polarization_group_box.setLayout(polarization_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.polarization_group_box)
        layout.addWidget(self.visualization_parameters_view)
        layout.addStretch()
        self.setLayout(layout)


class XMCDDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.difference_widget = VisualizationWidget('Difference')
        self.ratio_widget = VisualizationWidget('Ratio')
        self.sum_widget = VisualizationWidget('Sum')
        self.parameters_view = XMCDParametersView()
        self.status_bar = QStatusBar()

        contents_layout = QGridLayout()
        contents_layout.addWidget(self.difference_widget, 0, 0)
        contents_layout.addWidget(self.ratio_widget, 0, 1)
        contents_layout.addWidget(self.sum_widget, 1, 0)
        contents_layout.addWidget(self.parameters_view, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(contents_layout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
