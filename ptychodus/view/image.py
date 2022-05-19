from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QWheelEvent
from PyQt5.QtWidgets import QApplication, QCheckBox, QComboBox, QFormLayout, \
        QGraphicsPixmapItem, QGraphicsScene, QGraphicsSceneHoverEvent, \
        QGraphicsSceneMouseEvent, QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, \
        QLabel, QLineEdit, QPushButton, QSizePolicy, QSpinBox, QStyle, QVBoxLayout, QWidget

from .widgets import BottomTitledGroupBox, DecimalSlider


class ImageFileGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('File', parent)
        self.saveButton = QPushButton()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageFileGroupBox:
        view = cls(parent)

        pixmapi = getattr(QStyle, 'SP_DialogSaveButton')
        saveIcon = view.style().standardIcon(pixmapi)
        view.saveButton.setIcon(saveIcon)
        view.saveButton.setToolTip('Save Image')

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 50)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        return view


class ImageColormapGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Color Map', parent)
        self.complexComponentComboBox = QComboBox()
        self.scalarTransformComboBox = QComboBox()
        self.colormapComboBox = QComboBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageColormapGroupBox:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 50)
        layout.addWidget(view.complexComponentComboBox)
        layout.addWidget(view.scalarTransformComboBox)
        layout.addWidget(view.colormapComboBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        return view


class ImageDataRangeGroupBox(BottomTitledGroupBox):
    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Data Range', parent)
        self.vminSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.vmaxSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.autoButton = QPushButton('Auto')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageDataRangeGroupBox:
        view = cls(parent)

        layout = QFormLayout()
        layout.setContentsMargins(10, 10, 10, 50)
        layout.addRow('Min:', view.vminSlider)
        layout.addRow('Max:', view.vmaxSlider)
        layout.addRow(view.autoButton)
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

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 50)
        layout.addWidget(view.indexSpinBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)

        return view


class ImageRibbon(QWidget):
    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.imageFileGroupBox = ImageFileGroupBox.createInstance()
        self.colormapGroupBox = ImageColormapGroupBox.createInstance()
        self.dataRangeGroupBox = ImageDataRangeGroupBox.createInstance()
        self.indexGroupBox = IndexGroupBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageRibbon:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.imageFileGroupBox)
        layout.addWidget(view.colormapGroupBox)
        layout.addWidget(view.dataRangeGroupBox)
        layout.addWidget(view.indexGroupBox)
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
            app.setOverrideCursor(Qt.OpenHandCursor)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        app = QApplication.instance()

        if app:
            app.restoreOverrideCursor()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pass

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self.setPos(self.scenePos() + event.scenePos() - event.lastScenePos())

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pass


class ImageWidget(QGraphicsView):
    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self._pixmapItem = ImageItem()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageWidget:
        widget = cls(parent)

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


class ImageView(QWidget):
    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.imageRibbon = ImageRibbon.createInstance()
        self.imageWidget = ImageWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setMenuBar(view.imageRibbon)
        layout.addWidget(view.imageWidget)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        return view
