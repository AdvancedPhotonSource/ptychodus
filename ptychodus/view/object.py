from __future__ import annotations
from typing import Generic, Optional, TypeVar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGridLayout, QGroupBox, QLabel, QSizePolicy, QSpinBox, QVBoxLayout,
                             QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .widgets import DecimalSlider, LengthWidget, RepositoryTreeView

__all__ = [
    'ObjectEditorDialog',
    'ObjectParametersView',
    'ObjectView',
    'RandomObjectView',
]

T = TypeVar('T', bound=QWidget)


class ObjectParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        view.setLayout(layout)

        return view


class RandomObjectView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfLayersSpinBox = QSpinBox()
        self.layerDistanceWidget = LengthWidget.createInstance()
        self.extraPaddingXSpinBox = QSpinBox()
        self.extraPaddingYSpinBox = QSpinBox()
        self.amplitudeMeanSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.amplitudeDeviationSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.phaseDeviationSlider = DecimalSlider.createInstance(Qt.Horizontal)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> RandomObjectView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Layers:', view.numberOfLayersSpinBox)
        layout.addRow('Layer Distance:', view.layerDistanceWidget)
        layout.addRow('Extra Padding X:', view.extraPaddingXSpinBox)
        layout.addRow('Extra Padding Y:', view.extraPaddingYSpinBox)
        layout.addRow('Amplitude Mean:', view.amplitudeMeanSlider)
        layout.addRow('Amplitude Deviation:', view.amplitudeDeviationSlider)
        layout.addRow('Phase Deviation:', view.phaseDeviationSlider)
        view.setLayout(layout)

        return view


class CompareObjectParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.name1Label = QLabel('Name 1:')
        self.name1ComboBox = QComboBox()
        self.name2Label = QLabel('Name 2:')
        self.name2ComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> CompareObjectParametersView:
        view = cls(parent)
        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        layout = QGridLayout()
        layout.addWidget(view.name1Label, 0, 0)
        layout.addWidget(view.name1ComboBox, 0, 1)
        layout.addWidget(view.name2Label, 0, 2)
        layout.addWidget(view.name2ComboBox, 0, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        view.setLayout(layout)

        return view


class CompareObjectPlotView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> CompareObjectPlotView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view


class CompareObjectView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = CompareObjectParametersView.createInstance(self)
        self.plotView = CompareObjectPlotView.createInstance(self)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> CompareObjectView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.parametersView)
        layout.addWidget(view.plotView)
        view.setLayout(layout)

        return view


class ObjectEditorDialog(Generic[T], QDialog):

    def __init__(self, editorView: T, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.editorView = editorView
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       title: str,
                       editorView: T,
                       parent: Optional[QWidget] = None) -> ObjectEditorDialog[T]:
        view = cls(editorView, parent)
        view.setWindowTitle(title)

        view.buttonBox.addButton(QDialogButtonBox.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(editorView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.AcceptRole:
            self.accept()
        else:
            self.reject()


class ObjectView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = ObjectParametersView.createInstance()
        self.repositoryView = RepositoryTreeView.createInstance('Repository')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.repositoryView)
        view.setLayout(layout)

        return view
