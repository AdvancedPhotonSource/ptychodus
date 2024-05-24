from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QSpinBox, QStatusBar, QVBoxLayout, QWidget)

from .visualization import VisualizationParametersView, VisualizationWidget
from .widgets import LengthWidget


class ProbePropagationParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Parameters', parent)
        self.beginCoordinateWidget = LengthWidget.createInstance(isSigned=True)
        self.endCoordinateWidget = LengthWidget.createInstance(isSigned=True)
        self.numberOfStepsSpinBox = QSpinBox()
        self.visualizationParametersView = VisualizationParametersView.createInstance()

        propagationLayout = QFormLayout()
        propagationLayout.addRow('Begin Coordinate:', self.beginCoordinateWidget)
        propagationLayout.addRow('End Coordinate:', self.endCoordinateWidget)
        propagationLayout.addRow('Number of Steps:', self.numberOfStepsSpinBox)

        propagationGroupBox = QGroupBox('Propagation')
        propagationGroupBox.setLayout(propagationLayout)

        layout = QVBoxLayout()
        layout.addWidget(propagationGroupBox)
        layout.addWidget(self.visualizationParametersView)
        layout.addStretch()
        self.setLayout(layout)


class ProbePropagationDialog(QDialog):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.xyView = VisualizationWidget.createInstance('XY Plane')
        self.zxView = VisualizationWidget.createInstance('ZX Plane')
        self.parametersView = ProbePropagationParametersView()
        self.zyView = VisualizationWidget.createInstance('ZY Plane')
        self.propagateButton = QPushButton('Propagate')
        self.saveButton = QPushButton('Save')
        self.coordinateSlider = QSlider(Qt.Orientation.Horizontal)
        self.coordinateLabel = QLabel()
        self.statusBar = QStatusBar()

        actionLayout = QHBoxLayout()
        actionLayout.addWidget(self.propagateButton)
        actionLayout.addWidget(self.saveButton)

        coordinateLayout = QHBoxLayout()
        coordinateLayout.setContentsMargins(0, 0, 0, 0)
        coordinateLayout.addWidget(self.coordinateSlider)
        coordinateLayout.addWidget(self.coordinateLabel)

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(self.xyView, 0, 0)
        contentsLayout.addWidget(self.zxView, 0, 1)
        contentsLayout.addWidget(self.parametersView, 1, 0)
        contentsLayout.addWidget(self.zyView, 1, 1)
        contentsLayout.addLayout(actionLayout, 2, 0)
        contentsLayout.addLayout(coordinateLayout, 2, 1)
        contentsLayout.setColumnStretch(0, 1)
        contentsLayout.setColumnStretch(1, 2)

        layout = QVBoxLayout()
        layout.addLayout(contentsLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)
