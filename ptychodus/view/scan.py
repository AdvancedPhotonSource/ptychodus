from __future__ import annotations
from typing import Generic, Optional, TypeVar

from PyQt5.QtWidgets import (QAbstractButton, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QGroupBox, QLabel, QSpinBox, QVBoxLayout, QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .widgets import AngleWidget, LengthWidget, RepositoryTableView

__all__ = [
    'CartesianScanView',
    'ConcentricScanView',
    'LissajousScanView',
    'ScanEditorDialog',
    'ScanPlotView',
    'ScanTransformView',
    'ScanView',
    'SpiralScanView',
    'TabularScanView',
]

T = TypeVar('T', bound=QGroupBox)


class CartesianScanView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfPointsXSpinBox = QSpinBox()
        self.numberOfPointsYSpinBox = QSpinBox()
        self.stepSizeXWidget = LengthWidget.createInstance()
        self.stepSizeYWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> CartesianScanView:
        view = cls(parent)

        MAX_INT = 0x7FFFFFFF
        view.numberOfPointsXSpinBox.setRange(0, MAX_INT)
        view.numberOfPointsYSpinBox.setRange(0, MAX_INT)

        layout = QFormLayout()
        layout.addRow('Number Of Points X:', view.numberOfPointsXSpinBox)
        layout.addRow('Number Of Points Y:', view.numberOfPointsYSpinBox)
        layout.addRow('Step Size X:', view.stepSizeXWidget)
        layout.addRow('Step Size Y:', view.stepSizeYWidget)
        view.setLayout(layout)

        return view


class ConcentricScanView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfShellsSpinBox = QSpinBox()
        self.numberOfPointsInFirstShellSpinBox = QSpinBox()
        self.radialStepSizeWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ConcentricScanView:
        view = cls(parent)

        MAX_INT = 0x7FFFFFFF
        view.numberOfShellsSpinBox.setRange(0, MAX_INT)
        view.numberOfPointsInFirstShellSpinBox.setRange(0, MAX_INT)

        layout = QFormLayout()
        layout.addRow('Number Of Shells:', view.numberOfShellsSpinBox)
        layout.addRow('Number Of Points In First Shell:', view.numberOfPointsInFirstShellSpinBox)
        layout.addRow('Radial Step Size:', view.radialStepSizeWidget)
        view.setLayout(layout)

        return view


class SpiralScanView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfPointsSpinBox = QSpinBox()
        self.radiusScalarWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> SpiralScanView:
        view = cls(parent)

        MAX_INT = 0x7FFFFFFF
        view.numberOfPointsSpinBox.setRange(0, MAX_INT)

        layout = QFormLayout()
        layout.addRow('Number Of Points:', view.numberOfPointsSpinBox)
        layout.addRow('Radius Scalar:', view.radiusScalarWidget)
        view.setLayout(layout)

        return view


class LissajousScanView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfPointsSpinBox = QSpinBox()
        self.amplitudeXWidget = LengthWidget.createInstance()
        self.amplitudeYWidget = LengthWidget.createInstance()
        self.angularStepXWidget = AngleWidget.createInstance()
        self.angularStepYWidget = AngleWidget.createInstance()
        self.angularShiftWidget = AngleWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> LissajousScanView:
        view = cls(parent)

        MAX_INT = 0x7FFFFFFF
        view.numberOfPointsSpinBox.setRange(0, MAX_INT)

        layout = QFormLayout()
        layout.addRow('Number Of Points:', view.numberOfPointsSpinBox)
        layout.addRow('Amplitude X:', view.amplitudeXWidget)
        layout.addRow('Amplitude Y:', view.amplitudeYWidget)
        layout.addRow('Angular Step X:', view.angularStepXWidget)
        layout.addRow('Angular Step Y:', view.angularStepYWidget)
        layout.addRow('Angular Shift:', view.angularShiftWidget)
        view.setLayout(layout)

        return view


class TabularScanView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.label = QLabel('None.')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TabularScanView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.label)
        view.setLayout(layout)

        return view


class ScanTransformView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Transform', parent)
        self.transformComboBox = QComboBox()
        self.jitterRadiusWidget = LengthWidget.createInstance()
        self.centroidXCheckBox = QCheckBox('Centroid X:')
        self.centroidXWidget = LengthWidget.createInstance(isSigned=True)
        self.centroidYCheckBox = QCheckBox('Centroid Y:')
        self.centroidYWidget = LengthWidget.createInstance(isSigned=True)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanTransformView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('(x,y) \u2192', view.transformComboBox)
        layout.addRow('Jitter Radius:', view.jitterRadiusWidget)
        layout.addRow(view.centroidXCheckBox, view.centroidXWidget)
        layout.addRow(view.centroidYCheckBox, view.centroidYWidget)
        view.setLayout(layout)

        return view


class ScanEditorDialog(Generic[T], QDialog):

    def __init__(self, editorView: T, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = editorView
        self.transformView = ScanTransformView.createInstance()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       title: str,
                       editorView: T,
                       parent: Optional[QWidget] = None) -> ScanEditorDialog[T]:
        view = cls(editorView, parent)
        view.setWindowTitle(title)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(editorView)
        layout.addWidget(view.transformView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class ScanView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.repositoryView = RepositoryTableView.createInstance('Repository')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.repositoryView)
        view.setLayout(layout)

        return view


class ScanPlotView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ScanPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view
