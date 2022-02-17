from __future__ import annotations

from PyQt5.QtWidgets import *

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ReconstructorParametersView(QWidget):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.algorithmComboBox = QComboBox()
        self.reconstructorStackedWidget = QStackedWidget()
        self.reconstructButton = QPushButton('Reconstruct')

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ReconstructorParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Algorithm:', view.algorithmComboBox)
        layout.addRow(view.reconstructorStackedWidget)
        layout.addRow(view.reconstructButton)
        view.setLayout(layout)

        return view


class ReconstructorPlotView(QWidget):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        width = 1
        height = 1
        dpi = 200
        self.figure = Figure(figsize=(width, height), dpi = dpi)
        self.figureCanvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ReconstructorPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view

