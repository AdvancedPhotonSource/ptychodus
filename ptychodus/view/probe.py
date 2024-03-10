from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QFormLayout,
                             QGraphicsView, QGridLayout, QGroupBox, QPushButton, QSpinBox,
                             QStatusBar, QVBoxLayout, QWidget)

from .image import ImageView
from .widgets import DecimalSlider, LengthWidget


class ProbePropagationView(QGroupBox):

    def __init__(self, title: str, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.imageView = ImageView.createInstance(statusBar)

    @classmethod
    def createInstance(cls,
                       title: str,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> ProbePropagationView:
        view = cls(title, statusBar, parent)
        view.setAlignment(Qt.AlignHCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.imageView)
        view.setLayout(layout)

        return view


class ProbePropagationCrossSectionView(QGroupBox):

    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.graphicsView = QGraphicsView()

    @classmethod
    def createInstance(cls,
                       title: str,
                       parent: QWidget | None = None) -> ProbePropagationCrossSectionView:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.graphicsView)
        view.setLayout(layout)

        return view


class ProbePropagationParametersView(QGroupBox):

    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.startCoordinateWidget = LengthWidget.createInstance()
        self.stopCoordinateWidget = LengthWidget.createInstance()
        self.numberOfStepsSpinBox = QSpinBox()
        self.saveButton = QPushButton('Save')

    @classmethod
    def createInstance(cls,
                       title: str,
                       parent: QWidget | None = None) -> ProbePropagationParametersView:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        layout = QFormLayout()
        layout.addRow('Start Coordinate:', view.startCoordinateWidget)
        layout.addRow('Stop Coordinate:', view.stopCoordinateWidget)
        layout.addRow('Number of Steps:', view.numberOfStepsSpinBox)
        layout.addRow(view.saveButton)
        view.setLayout(layout)

        return view


class ProbePropagationDialog(QDialog):

    def __init__(self, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.xyView = ProbePropagationView.createInstance('XY Plane', statusBar)
        self.zxView = ProbePropagationCrossSectionView.createInstance('ZX Plane')
        self.parametersView = ProbePropagationParametersView.createInstance('Parameters')
        self.zyView = ProbePropagationCrossSectionView.createInstance('ZY Plane')
        self.propagateButton = QPushButton('Propagate')
        self.coordinateSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> ProbePropagationDialog:
        view = cls(statusBar, parent)
        view.setWindowTitle('Probe Propagation')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        contentsLayout = QGridLayout()
        contentsLayout.addWidget(view.xyView, 0, 0)
        contentsLayout.addWidget(view.zxView, 0, 1)
        contentsLayout.addWidget(view.parametersView, 1, 0)
        contentsLayout.addWidget(view.zyView, 1, 1)
        contentsLayout.addWidget(view.propagateButton, 2, 0)
        contentsLayout.addWidget(view.coordinateSlider, 2, 1)
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
