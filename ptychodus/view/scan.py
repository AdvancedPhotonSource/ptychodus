from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .widgets import LengthWidget


class ScanTransformView(QGroupBox):  # FIXME replace this

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Transform', parent)
        self.transformComboBox = QComboBox()
        self.jitterRadiusWidget = LengthWidget.createInstance()
        self.centroidXCheckBox = QCheckBox('Centroid X:')
        self.centroidXWidget = LengthWidget.createInstance(isSigned=True)
        self.centroidYCheckBox = QCheckBox('Centroid Y:')
        self.centroidYWidget = LengthWidget.createInstance(isSigned=True)

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ScanTransformView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('(x,y) \u2192', view.transformComboBox)
        layout.addRow('Jitter Radius:', view.jitterRadiusWidget)
        layout.addRow(view.centroidXCheckBox, view.centroidXWidget)
        layout.addRow(view.centroidYCheckBox, view.centroidYWidget)
        view.setLayout(layout)

        return view


class ScanPlotView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ScanPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
