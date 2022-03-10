from __future__ import annotations

from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox, QVBoxLayout, QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib

from .widgets import LengthWidget


class ScanScanView(QGroupBox):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__('Parameters', parent)
        self.initializerComboBox = QComboBox()
        self.numberOfScanPointsSpinBox = QSpinBox()
        self.extentXSpinBox = QSpinBox()
        self.extentYSpinBox = QSpinBox()
        self.stepSizeXWidget = LengthWidget.createInstance()
        self.stepSizeYWidget = LengthWidget.createInstance()
        self.jitterRadiusLineEdit = QLineEdit()
        self.transformComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ScanScanView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Initializer:', view.initializerComboBox)
        layout.addRow('Number of Points:', view.numberOfScanPointsSpinBox)
        layout.addRow('Extent X:', view.extentXSpinBox)
        layout.addRow('Extent Y:', view.extentYSpinBox)
        layout.addRow('Step Size X:', view.stepSizeXWidget)
        layout.addRow('Step Size Y:', view.stepSizeYWidget)
        layout.addRow('Jitter Radius [px]:', view.jitterRadiusLineEdit)
        layout.addRow('Transform (x,y) \u2192', view.transformComboBox)
        view.setLayout(layout)

        return view


class ScanParametersView(QWidget):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.scanView = ScanScanView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ScanParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.scanView)
        layout.addStretch()
        view.setLayout(layout)

        return view


class ScanPlotView(QWidget):
    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        width = 1
        height = 1
        dpi = 200
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figureCanvas = FigureCanvas(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> ScanPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
