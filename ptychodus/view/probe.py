from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout, QGridLayout,
                             QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
                             QVBoxLayout, QWidget)

from .visualization import VisualizationParametersView, VisualizationWidget
from .widgets import LengthWidget


class ProbePropagationParametersView(QGroupBox):

    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.beginCoordinateWidget = LengthWidget.createInstance()
        self.endCoordinateWidget = LengthWidget.createInstance()
        self.numberOfStepsSpinBox = QSpinBox()
        self.visualizationParametersView = VisualizationParametersView.createInstance()

    @classmethod
    def createInstance(cls,
                       title: str,
                       parent: QWidget | None = None) -> ProbePropagationParametersView:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        propagationLayout = QFormLayout()
        propagationLayout.addRow('Begin Coordinate:', view.beginCoordinateWidget)
        propagationLayout.addRow('End Coordinate:', view.endCoordinateWidget)
        propagationLayout.addRow('Number of Steps:', view.numberOfStepsSpinBox)

        propagationGroupBox = QGroupBox('Propagation')
        propagationGroupBox.setLayout(propagationLayout)

        layout = QVBoxLayout()
        layout.addWidget(propagationGroupBox)
        layout.addWidget(view.visualizationParametersView)
        layout.addStretch()
        view.setLayout(layout)

        return view


class ProbePropagationDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.xyView = VisualizationWidget.createInstance('XY Plane')
        self.zxView = VisualizationWidget.createInstance('ZX Plane')
        self.parametersView = ProbePropagationParametersView.createInstance('Parameters')
        self.zyView = VisualizationWidget.createInstance('ZY Plane')
        self.propagateButton = QPushButton('Propagate')
        self.saveButton = QPushButton('Save')
        self.coordinateSlider = QSlider(Qt.Orientation.Horizontal)
        self.coordinateLabel = QLabel()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ProbePropagationDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        actionLayout = QHBoxLayout()
        actionLayout.addWidget(view.propagateButton)
        actionLayout.addWidget(view.saveButton)

        coordinateLayout = QHBoxLayout()
        coordinateLayout.setContentsMargins(0, 0, 0, 0)
        coordinateLayout.addWidget(view.coordinateSlider)
        coordinateLayout.addWidget(view.coordinateLabel)

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(view.xyView, 0, 0)
        contentsLayout.addWidget(view.zxView, 0, 1)
        contentsLayout.addWidget(view.parametersView, 1, 0)
        contentsLayout.addWidget(view.zyView, 1, 1)
        contentsLayout.addLayout(actionLayout, 2, 0)
        contentsLayout.addLayout(coordinateLayout, 2, 1)
        contentsLayout.setColumnStretch(0, 1)
        contentsLayout.setColumnStretch(1, 2)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
