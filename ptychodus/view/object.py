from PyQt5.QtWidgets import (QComboBox, QDialog, QFormLayout, QGridLayout, QGroupBox, QLabel,
                             QPushButton, QStatusBar, QVBoxLayout, QWidget)

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
        self.lcircComboBox = QComboBox()
        self.rcircComboBox = QComboBox()
        self.saveButton = QPushButton('Save')
        self.visualizationParametersView = VisualizationParametersView.createInstance()

        polarizationLayout = QFormLayout()
        polarizationLayout.addRow('Left Circular:', self.lcircComboBox)
        polarizationLayout.addRow('Right Circular:', self.rcircComboBox)
        polarizationLayout.addRow(self.saveButton)
        self.polarizationGroupBox.setLayout(polarizationLayout)

        layout = QVBoxLayout()
        layout.addWidget(self.polarizationGroupBox)
        layout.addWidget(self.visualizationParametersView)
        layout.addStretch()
        self.setLayout(layout)


class XMCDDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.differenceWidget = VisualizationWidget.createInstance('Difference')
        self.ratioWidget = VisualizationWidget.createInstance('Ratio')
        self.sumWidget = VisualizationWidget.createInstance('Sum')
        self.parametersView = XMCDParametersView()
        self.statusBar = QStatusBar()

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(self.differenceWidget, 0, 0)
        contentsLayout.addWidget(self.ratioWidget, 0, 1)
        contentsLayout.addWidget(self.sumWidget, 1, 0)
        contentsLayout.addWidget(self.parametersView, 1, 1)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)
