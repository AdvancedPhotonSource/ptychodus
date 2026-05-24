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
