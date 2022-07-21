from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLineEdit, QPushButton, QScrollArea, \
        QStackedWidget, QVBoxLayout, QWidget

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class WorkflowDataView(QGroupBox):

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.uuidLineEdit = QLineEdit()
        self.pathLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, title: str, parent: Optional[QWidget] = None) -> WorkflowDataView:
        view = cls(title, parent)

        layout = QFormLayout()
        layout.addRow('Endpoint UUID:', view.uuidLineEdit)
        layout.addRow('Data Path:', view.pathLineEdit)
        view.setLayout(layout)

        return view


class WorkflowComputeView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Compute', parent)
        self.uuidLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowComputeView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Endpoint UUID:', view.uuidLineEdit)
        view.setLayout(layout)

        return view


class WorkflowParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.dataSourceView = WorkflowDataView.createInstance('Data Source')
        self.dataDestinationView = WorkflowDataView.createInstance('Data Destination')
        self.computeView = WorkflowComputeView.createInstance()
        self.launchButton = QPushButton('Launch')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.dataSourceView)
        layout.addWidget(view.dataDestinationView)
        layout.addWidget(view.computeView)
        layout.addWidget(view.launchButton)
        layout.addStretch()
        view.setLayout(layout)

        return view


class WorkflowPlotView(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> WorkflowPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
