from __future__ import annotations
from typing import Generic, TypeVar

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGridLayout, QGroupBox, QLabel, QSizePolicy, QSpinBox, QVBoxLayout,
                             QWidget)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .widgets import DecimalSlider, LengthWidget

__all__ = [
    'ObjectEditorDialog',
    'RandomObjectView',
]

T = TypeVar('T', bound=QWidget)


class RandomObjectView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Parameters', parent)
        self.numberOfLayersSpinBox = QSpinBox()
        self.layerDistanceWidget = LengthWidget.createInstance()
        self.extraPaddingXSpinBox = QSpinBox()
        self.extraPaddingYSpinBox = QSpinBox()
        self.amplitudeMeanSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.amplitudeDeviationSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.phaseDeviationSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> RandomObjectView:
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


class FourierRingCorrelationParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Parameters', parent)
        self.name1Label = QLabel('Name 1:')
        self.name1ComboBox = QComboBox()
        self.name2Label = QLabel('Name 2:')
        self.name2ComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FourierRingCorrelationParametersView:
        view = cls(parent)
        view.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

        layout = QGridLayout()
        layout.addWidget(view.name1Label, 0, 0)
        layout.addWidget(view.name1ComboBox, 0, 1)
        layout.addWidget(view.name2Label, 0, 2)
        layout.addWidget(view.name2ComboBox, 0, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        view.setLayout(layout)

        return view


class FourierRingCorrelationDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.parametersView = FourierRingCorrelationParametersView.createInstance()
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> FourierRingCorrelationDialog:
        view = cls(parent)
        view.setWindowTitle('Fourier Ring Correlation')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class ObjectEditorDialog(Generic[T], QDialog):

    def __init__(self, editorView: T, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.editorView = editorView
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       title: str,
                       editorView: T,
                       parent: QWidget | None = None) -> ObjectEditorDialog[T]:
        view = cls(editorView, parent)
        view.setWindowTitle(title)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(editorView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
