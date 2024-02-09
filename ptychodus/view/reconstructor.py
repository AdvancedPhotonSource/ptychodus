from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QLabel, QPushButton, QScrollArea,
                             QStackedWidget, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ReconstructorView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.algorithmLabel = QLabel('Algorithm:')
        self.algorithmComboBox = QComboBox()
        self.productLabel = QLabel('Product:')
        self.productComboBox = QComboBox()
        self.ingestButton = QPushButton('Ingest')
        self.saveButton = QPushButton('Save')
        self.trainButton = QPushButton('Train')
        self.clearButton = QPushButton('Clear')
        self.reconstructButton = QPushButton('Reconstruct')
        self.reconstructSplitButton = QPushButton('Split')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ReconstructorView:
        view = cls(parent)

        view.ingestButton.setToolTip('Ingest Training Data')
        view.saveButton.setToolTip('Save Training Data')
        view.trainButton.setToolTip('Train Model')
        view.clearButton.setToolTip('Reset Training Data Buffers')
        view.reconstructButton.setToolTip('Reconstruct Full Dataset')
        view.reconstructSplitButton.setToolTip('Reconstruct Odd/Even Split Dataset')

        layout = QGridLayout()
        layout.addWidget(view.algorithmLabel, 0, 0)
        layout.addWidget(view.algorithmComboBox, 0, 1, 1, 4)
        layout.addWidget(view.productLabel, 1, 0)
        layout.addWidget(view.productComboBox, 1, 1, 1, 4)
        layout.addWidget(view.ingestButton, 2, 1)
        layout.addWidget(view.saveButton, 2, 2)
        layout.addWidget(view.trainButton, 2, 3)
        layout.addWidget(view.clearButton, 2, 4)
        layout.addWidget(view.reconstructButton, 3, 1, 1, 3)
        layout.addWidget(view.reconstructSplitButton, 3, 4)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)
        view.setLayout(layout)

        return view


class ReconstructorParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.reconstructorView = ReconstructorView.createInstance()
        self.stackedWidget = QStackedWidget()
        self.scrollArea = QScrollArea()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ReconstructorParametersView:
        view = cls(parent)

        view.scrollArea.setWidgetResizable(True)
        view.scrollArea.setWidget(view.stackedWidget)

        view.stackedWidget.layout().setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.addWidget(view.reconstructorView)
        layout.addWidget(view.scrollArea)
        view.setLayout(layout)

        return view


class ReconstructorPlotView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ReconstructorPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
