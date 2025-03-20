from __future__ import annotations
from enum import auto, Enum
import logging

import numpy

from PyQt5.QtCore import pyqtSignal, Qt, QObject, QPointF, QLineF, QRectF, QSize, QSizeF
from PyQt5.QtGui import QIcon, QImage, QPalette, QPen, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QGroupBox,
    QLineEdit,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ptychodus.api.visualization import VisualizationProduct

from .widgets import DecimalLineEdit

logger = logging.getLogger(__name__)


class ImageItemEvents(QObject):
    rectangleFinished = pyqtSignal(QRectF)
    lineCutFinished = pyqtSignal(QLineF)


class ImageMouseTool(Enum):
    MOVE_TOOL = auto()
    RULER_TOOL = auto()
    RECTANGLE_TOOL = auto()
    LINE_CUT_TOOL = auto()


class ImageItem(QGraphicsPixmapItem):
    def __init__(self, events: ImageItemEvents, statusBar: QStatusBar) -> None:
        super().__init__()
        self._events = events
        self._statusBar = statusBar
        self._product: VisualizationProduct | None = None
        self._mouseTool = ImageMouseTool.MOVE_TOOL
        self._lineItem = QGraphicsLineItem(self)
        self._lineItem.hide()
        self._rectangleItem = QGraphicsRectItem(self)
        self._rectangleItem.hide()
        self._rectangleOrigin = QPointF()
        self.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)

    def getProduct(self) -> VisualizationProduct | None:
        return self._product

    def setProduct(self, product: VisualizationProduct) -> None:
        imageRGBAf = product.get_image_rgba()
        # NOTE .copy() ensures imageRGBAi is not a view
        imageRGBAi = numpy.multiply(imageRGBAf, 255).astype(numpy.uint8).copy()

        try:
            image = QImage(
                imageRGBAi.data,
                imageRGBAi.shape[1],
                imageRGBAi.shape[0],
                imageRGBAi.strides[0],
                QImage.Format.Format_RGBA8888,
            )
            pixmap = QPixmap.fromImage(image)
        except Exception as exc:
            logger.exception(exc)
            pixmap = QPixmap()

        self._product = product
        self.setPixmap(pixmap)

    def clearProduct(self) -> None:
        self._product = None
        self.setPixmap(QPixmap())

    def setMouseTool(self, mouseTool: ImageMouseTool) -> None:
        self._mouseTool = mouseTool

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        app = QApplication.instance()

        if app:
            cursor = Qt.CursorShape.CrossCursor

            if self._mouseTool == ImageMouseTool.MOVE_TOOL:
                cursor = Qt.CursorShape.OpenHandCursor

            app.setOverrideCursor(cursor)  # type: ignore

        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        pos = event.pos()

        if self._product is not None:
            infoText = self._product.get_info_text(pos.x(), pos.y())
            self._statusBar.showMessage(infoText)

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        app = QApplication.instance()

        if app:
            app.restoreOverrideCursor()  # type: ignore

        self._statusBar.clearMessage()
        super().hoverLeaveEvent(event)

    def _changeOverrideCursor(self, cursor: Qt.CursorShape) -> None:
        app = QApplication.instance()

        if app:
            app.changeOverrideCursor(cursor)  # type: ignore

    @staticmethod
    def _createPen(color: Qt.GlobalColor) -> QPen:
        pen = QPen(color)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        pen.setCosmetic(True)
        return pen

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mouseTool == ImageMouseTool.MOVE_TOOL:
            self._changeOverrideCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._mouseTool == ImageMouseTool.RULER_TOOL:
            line = QLineF(event.pos(), event.pos())
            self.prepareGeometryChange()
            self._lineItem.setLine(line)
            self._lineItem.setPen(self._createPen(Qt.GlobalColor.cyan))
            self._lineItem.show()
        elif self._mouseTool == ImageMouseTool.RECTANGLE_TOOL:
            self._rectangleOrigin = event.pos()
            rect = QRectF(self._rectangleOrigin, QSizeF())
            self.prepareGeometryChange()
            self._rectangleItem.setRect(rect)
            self._rectangleItem.setPen(self._createPen(Qt.GlobalColor.cyan))
            self._rectangleItem.show()
        elif self._mouseTool == ImageMouseTool.LINE_CUT_TOOL:
            line = QLineF(event.pos(), event.pos())
            self.prepareGeometryChange()
            self._lineItem.setLine(line)
            self._lineItem.setPen(self._createPen(Qt.GlobalColor.magenta))
            self._lineItem.show()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mouseTool == ImageMouseTool.MOVE_TOOL:
            self.setPos(self.scenePos() + event.scenePos() - event.lastScenePos())
        elif self._mouseTool == ImageMouseTool.RULER_TOOL:
            origin = self._lineItem.line().p1()
            line = QLineF(origin, event.pos())
            self.prepareGeometryChange()
            self._lineItem.setLine(line)
            message1 = f'{line.length():.1f} pixels, {line.angle():.2f}\u00b0'
            message2 = f'{line.dx():.1f} \u00d7 {line.dy():.1f}'
            self._statusBar.showMessage(f'{message1} ({message2})')
        elif self._mouseTool == ImageMouseTool.RECTANGLE_TOOL:
            rect = QRectF(self._rectangleOrigin, event.pos()).normalized()
            center = rect.center()
            self.prepareGeometryChange()
            self._rectangleItem.setRect(rect)
            message1 = f'{rect.width():.1f} \u00d7 {rect.height():.1f}'
            message2 = f'{center.x():.1f}, {center.y():.1f}'
            self._statusBar.showMessage(f'Rectangle: {message1} (Center: {message2})')
        elif self._mouseTool == ImageMouseTool.LINE_CUT_TOOL:
            origin = self._lineItem.line().p1()
            line = QLineF(origin, event.pos())
            self.prepareGeometryChange()
            self._lineItem.setLine(line)
            message1 = f'{line.length():.1f} pixels, {line.angle():.2f}\u00b0'
            message2 = f'{line.dx():.1f} \u00d7 {line.dy():.1f}'
            self._statusBar.showMessage(f'{message1} ({message2})')

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mouseTool == ImageMouseTool.MOVE_TOOL:
            self._changeOverrideCursor(Qt.CursorShape.OpenHandCursor)
        elif self._mouseTool == ImageMouseTool.RULER_TOOL:
            self._lineItem.setLine(QLineF())
            self._lineItem.hide()
        elif self._mouseTool == ImageMouseTool.RECTANGLE_TOOL:
            self._events.rectangleFinished.emit(self._rectangleItem.rect())
            self._rectangleItem.setRect(QRectF())
            self._rectangleItem.hide()
        elif self._mouseTool == ImageMouseTool.LINE_CUT_TOOL:
            self._events.lineCutFinished.emit(self._lineItem.line())
            self._lineItem.setLine(QLineF())
            self._lineItem.hide()


class LineCutDialog(QDialog):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> LineCutDialog:
        view = cls(parent)
        view.setWindowTitle('Line-Cut Dialog')

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas)
        view.setLayout(layout)

        return view


class RectangleView(QGroupBox):
    @staticmethod
    def _createReadOnlyLineEdit() -> QLineEdit:
        lineEdit = QLineEdit()

        palette = lineEdit.palette()
        palette.setColor(QPalette.Base, palette.color(QPalette.Window))
        palette.setColor(QPalette.Text, palette.color(QPalette.WindowText))
        lineEdit.setPalette(palette)

        lineEdit.setFocusPolicy(Qt.NoFocus)
        lineEdit.setReadOnly(True)

        return lineEdit

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Rectangle', parent)
        self.centerXLineEdit = RectangleView._createReadOnlyLineEdit()
        self.centerYLineEdit = RectangleView._createReadOnlyLineEdit()
        self.widthLineEdit = RectangleView._createReadOnlyLineEdit()
        self.heightLineEdit = RectangleView._createReadOnlyLineEdit()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> RectangleView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Center X:', view.centerXLineEdit)
        layout.addRow('Center Y:', view.centerYLineEdit)
        layout.addRow('Width:', view.widthLineEdit)
        layout.addRow('Height:', view.heightLineEdit)
        view.setLayout(layout)

        return view


class HistogramDialog(QDialog):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figureCanvas = FigureCanvasQTAgg(self.figure)
        self.navigationToolbar = NavigationToolbar(self.figureCanvas, self)
        self.axes = self.figure.add_subplot(111)
        self.rectangleView = RectangleView.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> HistogramDialog:
        view = cls(parent)
        view.setWindowTitle('Histogram')

        layout = QVBoxLayout()
        layout.addWidget(view.navigationToolbar)
        layout.addWidget(view.figureCanvas, 1)
        layout.addWidget(view.rectangleView)
        view.setLayout(layout)

        return view


class VisualizationView(QGraphicsView):
    def wheelEvent(self, event: QWheelEvent) -> None:
        oldPosition = self.mapToScene(event.pos())

        zoomBase = 1.25
        zoom = zoomBase if event.angleDelta().y() > 0 else 1.0 / zoomBase
        self.scale(zoom, zoom)

        newPosition = self.mapToScene(event.pos())

        deltaPosition = newPosition - oldPosition
        self.translate(deltaPosition.x(), deltaPosition.y())


class VisualizationWidget(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.toolBar = QToolBar('Tools')
        self.homeAction = QAction(QIcon(':/icons/home'), 'Home')
        self.saveAction = QAction(QIcon(':/icons/save'), 'Save Image')
        self.autoscaleAction = QAction(QIcon(':/icons/autoscale'), 'Autoscale Color Axis')
        self.visualizationView = VisualizationView()

    @classmethod
    def create_instance(cls, title: str, parent: QWidget | None = None) -> VisualizationWidget:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        view.toolBar.setFloatable(False)
        view.toolBar.setIconSize(QSize(32, 32))
        view.toolBar.addAction(view.homeAction)
        view.toolBar.addAction(view.saveAction)
        view.toolBar.addAction(view.autoscaleAction)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.toolBar)
        layout.addWidget(view.visualizationView)
        view.setLayout(layout)

        return view


class VisualizationParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Visualization', parent)

        self.rendererComboBox = QComboBox()
        self.transformationComboBox = QComboBox()
        self.variantComboBox = QComboBox()
        self.minDisplayValueLineEdit = DecimalLineEdit.create_instance()
        self.maxDisplayValueLineEdit = DecimalLineEdit.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> VisualizationParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Renderer:', view.rendererComboBox)
        layout.addRow('Transform:', view.transformationComboBox)
        layout.addRow('Variant:', view.variantComboBox)
        layout.addRow('Min Display Value:', view.minDisplayValueLineEdit)
        layout.addRow('Max Display Value:', view.maxDisplayValueLineEdit)
        view.setLayout(layout)

        return view
