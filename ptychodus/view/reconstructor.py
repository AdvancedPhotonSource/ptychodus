from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QComboBox, QGridLayout, QGroupBox, QLabel, QPushButton, QScrollArea,
                             QStackedWidget, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ReconstructorView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.algorithmLabel = QLabel('Algorithm:')
        self.algorithmComboBox = QComboBox()
        self.scanLabel = QLabel('Scan:')
        self.scanComboBox = QComboBox()
        self.scanValidationLabel = QLabel()
        self.probeLabel = QLabel('Probe:')
        self.probeComboBox = QComboBox()
        self.probeValidationLabel = QLabel()
        self.objectLabel = QLabel('Object:')
        self.objectComboBox = QComboBox()
        self.objectValidationLabel = QLabel()
        self.reconstructButton = QPushButton('Reconstruct')
        self.ingestButton = QPushButton('Ingest')
        self.trainButton = QPushButton('Train')
        self.resetButton = QPushButton('Reset')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ReconstructorView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.algorithmLabel, 0, 0)
        layout.addWidget(view.algorithmComboBox, 0, 1, 1, 3)
        layout.addWidget(view.scanLabel, 1, 0)
        layout.addWidget(view.scanComboBox, 1, 1, 1, 3)
        layout.addWidget(view.scanValidationLabel, 1, 4)
        layout.addWidget(view.probeLabel, 2, 0)
        layout.addWidget(view.probeComboBox, 2, 1, 1, 3)
        layout.addWidget(view.probeValidationLabel, 2, 4)
        layout.addWidget(view.objectLabel, 3, 0)
        layout.addWidget(view.objectComboBox, 3, 1, 1, 3)
        layout.addWidget(view.objectValidationLabel, 3, 4)
        layout.addWidget(view.reconstructButton, 4, 1, 1, 3)
        layout.addWidget(view.ingestButton, 5, 1)
        layout.addWidget(view.trainButton, 5, 2)
        layout.addWidget(view.resetButton, 5, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
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
        self.figureCanvas = FigureCanvas(self.figure)
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
