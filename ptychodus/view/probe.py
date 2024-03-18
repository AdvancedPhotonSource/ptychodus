from __future__ import annotations

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (QAbstractButton, QAction, QComboBox, QDialog, QDialogButtonBox,
                             QFormLayout, QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QSpinBox, QStatusBar, QToolBar,
                             QVBoxLayout, QWidget)

from .image import ImageItem, ImageItemEvents
from .widgets import DecimalLineEdit, LengthWidget


class ProbePropagationCrossSectionView(QGroupBox):

    def __init__(self, title: str, imageItem: ImageItem, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self._imageItem = imageItem
        self.toolBar = QToolBar('Tools')
        self.homeAction = QAction(QIcon(':/icons/home'), 'Home')
        self.saveAction = QAction(QIcon(':/icons/save'), 'Save Image')
        self.graphicsView = QGraphicsView()

    @classmethod
    def createInstance(cls,
                       title: str,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> ProbePropagationCrossSectionView:
        imageItemEvents = ImageItemEvents()
        imageItem = ImageItem(imageItemEvents, statusBar)

        view = cls(title, imageItem, parent)
        view.setAlignment(Qt.AlignHCenter)

        view.toolBar.setFloatable(False)
        view.toolBar.setIconSize(QSize(32, 32))
        view.toolBar.addAction(view.homeAction)
        view.toolBar.addAction(view.saveAction)

        view.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.toolBar)
        layout.addWidget(view.graphicsView)
        view.setLayout(layout)

        return view

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._imageItem.setPixmap(pixmap)

    def getPixmap(self) -> QPixmap:
        return self._imageItem.pixmap()

    def zoomToFit(self) -> None:
        self._imageItem.setPos(0, 0)
        scene = self.graphicsView.scene()
        boundingRect = scene.itemsBoundingRect()
        scene.setSceneRect(boundingRect)
        self.graphicsView.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        oldPosition = self.graphicsView.mapToScene(event.pos())

        zoomBase = 1.25
        zoom = zoomBase if event.angleDelta().y() > 0 else 1. / zoomBase
        self.graphicsView.scale(zoom, zoom)

        newPosition = self.graphicsView.mapToScene(event.pos())

        deltaPosition = newPosition - oldPosition
        self.graphicsView.translate(deltaPosition.x(), deltaPosition.y())


class DisplayRangeWidget(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.minValueLineEdit = DecimalLineEdit.createInstance()
        self.maxValueLineEdit = DecimalLineEdit.createInstance()
        self.autoButton = QPushButton('Auto')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> DisplayRangeWidget:
        widget = DisplayRangeWidget(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget.minValueLineEdit)
        layout.addWidget(widget.maxValueLineEdit)
        layout.addWidget(widget.autoButton)
        widget.setLayout(layout)

        return widget


class ProbePropagationParametersView(QGroupBox):

    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)

        self.propagationGroupBox = QGroupBox('Propagation')
        self.startCoordinateWidget = LengthWidget.createInstance()
        self.stopCoordinateWidget = LengthWidget.createInstance()
        self.numberOfStepsSpinBox = QSpinBox()

        self.visualizationGroupBox = QGroupBox('Visualization')
        self.colorizerComboBox = QComboBox()
        self.scalarTransformComboBox = QComboBox()
        self.variantComboBox = QComboBox()
        self.displayRangeWidget = DisplayRangeWidget.createInstance()

    @classmethod
    def createInstance(cls,
                       title: str,
                       parent: QWidget | None = None) -> ProbePropagationParametersView:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        propagationLayout = QFormLayout()
        propagationLayout.addRow('Start Coordinate:', view.startCoordinateWidget)
        propagationLayout.addRow('Stop Coordinate:', view.stopCoordinateWidget)
        propagationLayout.addRow('Number of Steps:', view.numberOfStepsSpinBox)
        view.propagationGroupBox.setLayout(propagationLayout)

        visualizationLayout = QFormLayout()
        visualizationLayout.addRow('Colorizer:', view.colorizerComboBox)
        visualizationLayout.addRow('Transform:', view.scalarTransformComboBox)
        visualizationLayout.addRow('Variant:', view.variantComboBox)
        visualizationLayout.addRow('Display Range:', view.displayRangeWidget)
        view.visualizationGroupBox.setLayout(visualizationLayout)

        layout = QVBoxLayout()
        layout.addWidget(view.propagationGroupBox)
        layout.addWidget(view.visualizationGroupBox)
        layout.addStretch()
        view.setLayout(layout)

        return view


class ProbePropagationDialog(QDialog):

    def __init__(self, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.xyView = ProbePropagationCrossSectionView.createInstance('XY Plane', statusBar)
        self.zxView = ProbePropagationCrossSectionView.createInstance('ZX Plane', statusBar)
        self.parametersView = ProbePropagationParametersView.createInstance('Parameters')
        self.zyView = ProbePropagationCrossSectionView.createInstance('ZY Plane', statusBar)
        self.propagateButton = QPushButton('Propagate')
        self.saveButton = QPushButton('Save')
        self.coordinateSlider = QSlider(Qt.Orientation.Horizontal)
        self.coordinateLabel = QLabel()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> ProbePropagationDialog:
        view = cls(statusBar, parent)

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
