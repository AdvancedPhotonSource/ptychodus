from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

from .widgets import LengthWidget


class ScanScanView(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Parameters', parent)
        self.numberOfScanPointsSpinBox = QSpinBox()
        self.extentXSpinBox = QSpinBox()
        self.extentYSpinBox = QSpinBox()
        self.stepSizeXWidget = LengthWidget.createInstance()
        self.stepSizeYWidget = LengthWidget.createInstance()
        self.jitterRadiusWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanScanView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Points:', view.numberOfScanPointsSpinBox)
        layout.addRow('Extent X:', view.extentXSpinBox)
        layout.addRow('Extent Y:', view.extentYSpinBox)
        layout.addRow('Step Size X:', view.stepSizeXWidget)
        layout.addRow('Step Size Y:', view.stepSizeYWidget)
        layout.addRow('Jitter Radius:', view.jitterRadiusWidget)
        view.setLayout(layout)

        return view


class ScanInitializerView(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Initializer', parent)
        self.initializerComboBox = QComboBox()
        self.initializeButton = QPushButton('Initialize')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanInitializerView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerComboBox)
        layout.addWidget(view.initializeButton)
        view.setLayout(layout)

        return view


class ScanTransformView(QGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Transform', parent)
        self.transformComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanTransformView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('(x,y) \u2192', view.transformComboBox)
        view.setLayout(layout)

        return view


class ScanParametersView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.initializerView = ScanInitializerView.createInstance()
        self.scanView = ScanScanView.createInstance()
        self.transformView = ScanTransformView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerView)
        layout.addWidget(view.scanView)
        layout.addWidget(view.transformView)
        layout.addStretch()
        view.setLayout(layout)

        return view


class ScanPlotView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
