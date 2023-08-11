from __future__ import annotations
import math
from decimal import Decimal
from typing import Optional
from PyQt5 import QtGui

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon, QPixmap, QWheelEvent, QMouseEvent
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGraphicsPixmapItem, QGraphicsScene, QGraphicsSceneHoverEvent,
                             QGraphicsSceneMouseEvent, QGraphicsView, QHBoxLayout, QPushButton,
                             QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget, QStatusBar)

from .widgets import BottomTitledGroupBox, DecimalLineEdit, DecimalSlider


class ImageDisplayRangeDialog(QDialog):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.buttonBox = QDialogButtonBox()
        self.minValueLineEdit = DecimalLineEdit.createInstance()
        self.maxValueLineEdit = DecimalLineEdit.createInstance()

    def setMinAndMaxValues(self, minValue: Decimal, maxValue: Decimal) -> None:
        self.minValueLineEdit.setValue(minValue)
        self.maxValueLineEdit.setValue(maxValue)

    def minValue(self) -> Decimal:
        return self.minValueLineEdit.getValue()

    def maxValue(self) -> Decimal:
        return self.maxValueLineEdit.getValue()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageDisplayRangeDialog:
        dialog = cls(parent)
        dialog.setWindowTitle('Set Display Range')
        dialog.buttonBox.addButton(QDialogButtonBox.Ok)
        dialog.buttonBox.accepted.connect(dialog.accept)
        dialog.buttonBox.addButton(QDialogButtonBox.Cancel)
        dialog.buttonBox.rejected.connect(dialog.reject)

        layout = QFormLayout()
        layout.addRow('Minimum Displayed Value:', dialog.minValueLineEdit)
        layout.addRow('Maximum Displayed Value:', dialog.maxValueLineEdit)
        layout.addRow(dialog.buttonBox)
        dialog.setLayout(layout)

        return dialog


class ImageFileGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('File', parent)
        self.saveButton = QToolButton()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageFileGroupBox:
        view = cls(parent)

        view.saveButton.setIcon(QIcon(':/icons/save'))
        view.saveButton.setIconSize(QSize(32, 32))
        view.saveButton.setToolTip('Save Image')

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        return view


class ImageColorizerGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Colorize', parent)
        self.colorizerComboBox = QComboBox()
        self.scalarTransformComboBox = QComboBox()
        self.variantComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageColorizerGroupBox:
        view = cls(parent)

        view.colorizerComboBox.setToolTip('Array Component')
        view.scalarTransformComboBox.setToolTip('Scalar Transform')
        view.variantComboBox.setToolTip('Colorizer')

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addWidget(view.colorizerComboBox)
        layout.addWidget(view.scalarTransformComboBox)
        layout.addWidget(view.variantComboBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        return view


class ImageDataRangeGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Data Range', parent)
        self.minDisplayValueSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.maxDisplayValueSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.autoButton = QPushButton('Auto')
        self.setButton = QPushButton('Set')
        self.displayRangeDialog = ImageDisplayRangeDialog.createInstance(self)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageDataRangeGroupBox:
        view = cls(parent)

        view.minDisplayValueSlider.setToolTip('Minimum Display Value')
        view.maxDisplayValueSlider.setToolTip('Maximum Display Value')
        view.autoButton.setToolTip('Rescale to Data Range')
        view.setButton.setToolTip('Rescale to Custom Range')

        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.addWidget(view.autoButton)
        buttonLayout.addWidget(view.setButton)

        layout = QFormLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addRow('Min:', view.minDisplayValueSlider)
        layout.addRow('Max:', view.maxDisplayValueSlider)
        layout.addRow(buttonLayout)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        return view


class IndexGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Index', parent)
        self.indexSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> IndexGroupBox:
        view = cls(parent)

        view.indexSpinBox.setToolTip('Image Index')

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 35)
        layout.addWidget(view.indexSpinBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        return view


class ImageRibbon(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.imageFileGroupBox = ImageFileGroupBox.createInstance()
        self.colormapGroupBox = ImageColorizerGroupBox.createInstance()
        self.dataRangeGroupBox = ImageDataRangeGroupBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageRibbon:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.imageFileGroupBox)
        layout.addWidget(view.colormapGroupBox)
        layout.addWidget(view.dataRangeGroupBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        return view


class ImageItem(QGraphicsPixmapItem):

    def __init__(self) -> None:
        super().__init__()
        self.setTransformationMode(Qt.FastTransformation)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        app = QApplication.instance()

        if app:
            app.setOverrideCursor(Qt.OpenHandCursor)  # type: ignore

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        app = QApplication.instance()

        if app:
            app.restoreOverrideCursor()  # type: ignore

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pass

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.setPos(self.scenePos() + event.scenePos() - event.lastScenePos())

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pass


class ImageWidget(QGraphicsView):

    def __init__(self, statusbar: QStatusBar, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self._pixmapItem = ImageItem()
        self._statusbar = statusbar

    @classmethod
    def createInstance(cls, statusbar: QStatusBar, parent: Optional[QWidget] = None) -> ImageWidget:
        widget = cls(statusbar, parent)

        scene = QGraphicsScene()
        scene.addItem(widget._pixmapItem)
        widget.setScene(scene)

        return widget

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._pixmapItem.setPixmap(pixmap)

    def getPixmap(self) -> QPixmap:
        return self._pixmapItem.pixmap()

    def wheelEvent(self, event: QWheelEvent) -> None:
        oldPosition = self.mapToScene(event.pos())

        zoomBase = 1.25
        zoom = zoomBase if event.angleDelta().y() > 0 else 1. / zoomBase
        self.scale(zoom, zoom)

        newPosition = self.mapToScene(event.pos())

        deltaPosition = newPosition - oldPosition
        self.translate(deltaPosition.x(), deltaPosition.y())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        scene_pos = self.mapToScene(event.pos())
        item = self.itemAt(scene_pos.toPoint())
        if isinstance(item, QGraphicsPixmapItem):
            pixel_coords = item.mapFromScene(scene_pos)
            self._statusbar.showMessage(
                'Mouse click coordinates: '
                f'{math.floor(pixel_coords.x())},{math.floor(pixel_coords.y())}'
            )
        super().mousePressEvent(event)

class ImageView(QWidget):

    def __init__(self, statusbar: QStatusBar, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.imageRibbon = ImageRibbon.createInstance()
        self.imageWidget = ImageWidget.createInstance(statusbar)

    @classmethod
    def createInstance(cls, statusbar: QStatusBar, parent: Optional[QWidget] = None) -> ImageView:
        view = cls(statusbar, parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setMenuBar(view.imageRibbon)
        layout.addWidget(view.imageWidget)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        return view
