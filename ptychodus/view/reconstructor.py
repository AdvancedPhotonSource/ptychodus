from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class ReconstructorView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Reconstructor', parent)
        self.reconstructorComboBox = QComboBox()
        self.reconstructButton = QPushButton('Reconstruct')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ReconstructorView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.reconstructorComboBox)
        layout.addWidget(view.reconstructButton)
        view.setLayout(layout)

        return view


class ReconstructorParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.reconstructorView = ReconstructorView.createInstance()
        self.reconstructorStackedWidget = QStackedWidget()

    @property
    def algorithmComboBox(self) -> QComboBox:
        return self.reconstructorView.reconstructorComboBox

    @property
    def reconstructButton(self) -> QPushButton:
        return self.reconstructorView.reconstructButton

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ReconstructorParametersView:
        view = cls(parent)

        view.reconstructorStackedWidget.layout().setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.addWidget(view.reconstructorView)
        layout.addWidget(view.reconstructorStackedWidget)
        view.setLayout(layout)

        return view


class ReconstructorPlotView(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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
