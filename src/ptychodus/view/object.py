from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .visualization import VisualizationParametersView, VisualizationWidget


class FourierRingCorrelationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product1Label = QLabel('Product 1:')
        self.product1ComboBox = QComboBox()
        self.product2Label = QLabel('Product 2:')
        self.product2ComboBox = QComboBox()
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

        parametersLayout = QGridLayout()
        parametersLayout.addWidget(self.product1Label, 0, 0)
        parametersLayout.addWidget(self.product1ComboBox, 0, 1)
        parametersLayout.addWidget(self.product2Label, 0, 2)
        parametersLayout.addWidget(self.product2ComboBox, 0, 3)
        parametersLayout.setColumnStretch(1, 1)
        parametersLayout.setColumnStretch(3, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.navigationToolbar)
        layout.addWidget(self.figureCanvas)
        layout.addLayout(parametersLayout)
        self.setLayout(layout)


class XMCDParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)

        self.polarizationGroupBox = QGroupBox('Polarization')
        self.lcirc_combo_box = QComboBox()
        self.rcirc_combo_box = QComboBox()
        self.save_button = QPushButton('Save')
        self.visualization_parameters_view = VisualizationParametersView.create_instance()

        polarizationLayout = QFormLayout()
        polarizationLayout.addRow('Left Circular:', self.lcirc_combo_box)
        polarizationLayout.addRow('Right Circular:', self.rcirc_combo_box)
        polarizationLayout.addRow(self.save_button)
        self.polarizationGroupBox.setLayout(polarizationLayout)

        layout = QVBoxLayout()
        layout.addWidget(self.polarizationGroupBox)
        layout.addWidget(self.visualization_parameters_view)
        layout.addStretch()
        self.setLayout(layout)


class XMCDDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.difference_widget = VisualizationWidget.create_instance('Difference')
        self.ratio_widget = VisualizationWidget.create_instance('Ratio')
        self.sum_widget = VisualizationWidget.create_instance('Sum')
        self.parameters_view = XMCDParametersView()
        self.status_bar = QStatusBar()

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(self.difference_widget, 0, 0)
        contentsLayout.addWidget(self.ratio_widget, 0, 1)
        contentsLayout.addWidget(self.sum_widget, 1, 0)
        contentsLayout.addWidget(self.parameters_view, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.status_bar)
        self.setLayout(layout)
