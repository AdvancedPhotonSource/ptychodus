from __future__ import annotations
from collections.abc import Generator
from decimal import Decimal
from enum import auto, Enum
from typing import Optional

from PyQt5.QtCore import Qt, QPoint, QPointF, QLineF, QRect, QRectF, QSize, QSizeF
from PyQt5.QtGui import (QColor, QConicalGradient, QIcon, QLinearGradient, QPainter, QPen, QPixmap,
                         QWheelEvent)
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGraphicsLineItem, QGraphicsPixmapItem, QGraphicsRectItem,
                             QGraphicsScene, QGraphicsSceneHoverEvent, QGraphicsSceneMouseEvent,
                             QGraphicsView, QGridLayout, QHBoxLayout, QPushButton, QSizePolicy,
                             QSpinBox, QStatusBar, QToolButton, QVBoxLayout, QWidget)

from ..api.image import RealArrayType
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


class ImageToolsGroupBox(BottomTitledGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Tools', parent)
        self.homeButton = QToolButton()
        self.saveButton = QToolButton()
        self.moveButton = QToolButton()
        self.rulerButton = QToolButton()
        self.rectangleButton = QToolButton()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageToolsGroupBox:
        view = cls(parent)

        view.homeButton.setIcon(QIcon(':/icons/home'))
        view.homeButton.setIconSize(QSize(48, 48))
        view.homeButton.setToolTip('Home')

        view.saveButton.setIcon(QIcon(':/icons/save'))
        view.saveButton.setIconSize(QSize(48, 48))
        view.saveButton.setToolTip('Save Image')

        view.moveButton.setIcon(QIcon(':/icons/move'))
        view.moveButton.setIconSize(QSize(32, 32))
        view.moveButton.setToolTip('Move')

        view.rulerButton.setIcon(QIcon(':/icons/ruler'))
        view.rulerButton.setIconSize(QSize(32, 32))
        view.rulerButton.setToolTip('Ruler')

        view.rectangleButton.setIcon(QIcon(':/icons/rectangle'))
        view.rectangleButton.setIconSize(QSize(32, 32))
        view.rectangleButton.setToolTip('Rectangle')

        layout = QGridLayout()
        layout.addWidget(view.homeButton, 0, 0, 1, 3)
        layout.setAlignment(view.homeButton, Qt.AlignHCenter)
        layout.addWidget(view.saveButton, 0, 3, 1, 3)
        layout.setAlignment(view.saveButton, Qt.AlignHCenter)
        layout.addWidget(view.moveButton, 1, 0, 1, 2)
        layout.addWidget(view.rulerButton, 1, 2, 1, 2)
        layout.addWidget(view.rectangleButton, 1, 4, 1, 2)
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
        self.editButton = QPushButton('Edit')
        self.colorLegendButton = QPushButton('Color Legend')
        self.displayRangeDialog = ImageDisplayRangeDialog.createInstance(self)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageDataRangeGroupBox:
        view = cls(parent)

        view.minDisplayValueSlider.setToolTip('Minimum Display Value')
        view.maxDisplayValueSlider.setToolTip('Maximum Display Value')
        view.autoButton.setToolTip('Rescale to Data Range')
        view.editButton.setToolTip('Rescale to Custom Range')
        view.colorLegendButton.setToolTip('Toggle Color Legend Visibility')

        buttonLayout = QHBoxLayout()
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.addWidget(view.autoButton)
        buttonLayout.addWidget(view.editButton)
        buttonLayout.addWidget(view.colorLegendButton)

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
        self.imageToolsGroupBox = ImageToolsGroupBox.createInstance()
        self.colormapGroupBox = ImageColorizerGroupBox.createInstance()
        self.dataRangeGroupBox = ImageDataRangeGroupBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageRibbon:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.imageToolsGroupBox)
        layout.addWidget(view.colormapGroupBox)
        layout.addWidget(view.dataRangeGroupBox)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        return view


class ImageMouseTool(Enum):
    MOVE_TOOL = auto()
    RULER_TOOL = auto()
    RECTANGLE_TOOL = auto()


class ImageItem(QGraphicsPixmapItem):

    def __init__(self, statusBar: QStatusBar) -> None:
        super().__init__()
        self._statusBar = statusBar
        self._mouseTool = ImageMouseTool.MOVE_TOOL
        self._rulerItem = QGraphicsLineItem(self)
        self._rulerItem.hide()
        self._rectangleItem = QGraphicsRectItem(self)
        self._rectangleItem.hide()
        self._rectangleOrigin = QPointF()
        self.setTransformationMode(Qt.FastTransformation)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)

    def setMouseTool(self, mouseTool: ImageMouseTool) -> None:
        self._mouseTool = mouseTool

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        app = QApplication.instance()

        if app:
            cursor = Qt.CrossCursor

            if self._mouseTool == ImageMouseTool.MOVE_TOOL:
                cursor = Qt.OpenHandCursor

            app.setOverrideCursor(cursor)  # type: ignore

        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        pos = event.pos()
        self._statusBar.showMessage(f'{pos.x():.1f}, {pos.y():.1f}')
        # FIXME display value
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

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mouseTool == ImageMouseTool.MOVE_TOOL:
            self._changeOverrideCursor(Qt.ClosedHandCursor)
        elif self._mouseTool == ImageMouseTool.RULER_TOOL:
            line = QLineF(event.pos(), event.pos())
            self.prepareGeometryChange()
            self._rulerItem.setLine(line)
            self._rulerItem.show()
        elif self._mouseTool == ImageMouseTool.RECTANGLE_TOOL:
            self._rectangleOrigin = event.pos()
            rect = QRectF(self._rectangleOrigin, QSizeF())
            self.prepareGeometryChange()
            self._rectangleItem.setRect(rect)
            self._rectangleItem.show()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mouseTool == ImageMouseTool.MOVE_TOOL:
            self.setPos(self.scenePos() + event.scenePos() - event.lastScenePos())
        elif self._mouseTool == ImageMouseTool.RULER_TOOL:
            origin = self._rulerItem.line().p1()
            line = QLineF(origin, event.pos())
            self.prepareGeometryChange()
            self._rulerItem.setLine(line)
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

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._mouseTool == ImageMouseTool.MOVE_TOOL:
            self._changeOverrideCursor(Qt.OpenHandCursor)
        elif self._mouseTool == ImageMouseTool.RULER_TOOL:
            self._rulerItem.setLine(QLineF())
            self._rulerItem.hide()
        elif self._mouseTool == ImageMouseTool.RECTANGLE_TOOL:
            self._rectangleItem.setRect(QRectF())
            self._rectangleItem.hide()


class ImageWidget(QGraphicsView):

    def __init__(self, imageItem: ImageItem, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self._imageItem = imageItem
        self._colorLegendMinValue = 0.
        self._colorLegendMaxValue = 1.
        self._colorLegendStopPoints: list[tuple[float, QColor]] = [
            (0.0, QColor(Qt.green)),
            (0.5, QColor(Qt.yellow)),
            (1.0, QColor(Qt.red)),
        ]
        self._colorLegendNumberOfTicks = 5  # TODO
        self._isColorLegendVisible = False
        self._isColorLegendCyclic = False

    @classmethod
    def createInstance(cls,
                       statusBar: QStatusBar,
                       parent: Optional[QWidget] = None) -> ImageWidget:
        imageItem = ImageItem(statusBar)
        widget = cls(imageItem, parent)
        widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scene = QGraphicsScene()
        scene.addItem(imageItem)
        widget.setScene(scene)

        return widget

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._imageItem.setPixmap(pixmap)

    def getPixmap(self) -> QPixmap:
        return self._imageItem.pixmap()

    def setMouseTool(self, mouseTool: ImageMouseTool) -> None:
        self._imageItem.setMouseTool(mouseTool)

    def _forceRedraw(self) -> None:
        self.scene().update()

    def setColorLegendColors(self, xArray: RealArrayType, rgbaArray: RealArrayType,
                             isCyclic: bool) -> None:
        colorLegendStopPoints: list[tuple[float, QColor]] = list()

        for x, rgba in zip(xArray, rgbaArray):
            color = QColor()
            color.setRgbF(rgba[0], rgba[1], rgba[2], rgba[3])
            colorLegendStopPoints.append((x, color))

        self._colorLegendStopPoints = colorLegendStopPoints
        self._isColorLegendCyclic = isCyclic
        self._forceRedraw()

    def setColorLegendRange(self, minValue: float, maxValue: float) -> None:
        self._colorLegendMinValue = minValue
        self._colorLegendMaxValue = maxValue
        self._forceRedraw()

    def setColorLegendVisible(self, visible: bool):
        self._isColorLegendVisible = visible
        self._forceRedraw()

    def zoomToFit(self) -> None:
        self._imageItem.setPos(0, 0)
        scene = self.scene()
        boundingRect = scene.itemsBoundingRect()
        scene.setSceneRect(boundingRect)
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    @property
    def _colorLegendTicks(self) -> Generator[float, None, None]:
        for tick in range(self._colorLegendNumberOfTicks):
            a = tick / (self._colorLegendNumberOfTicks - 1)
            yield (1. - a) * self._colorLegendMinValue + a * self._colorLegendMaxValue

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if not self._isColorLegendVisible:
            return

        fgPainter = QPainter(self.viewport())

        # cosmetic pens have the same thickness at different scale factors
        pen = QPen()
        pen.setCosmetic(True)
        pen.setWidth(3)
        fgPainter.setPen(pen)

        fontMetrics = fgPainter.fontMetrics()
        dx = fontMetrics.horizontalAdvance('m')
        dy = fontMetrics.lineSpacing()

        widgetRect = self.viewport().rect()

        if self._isColorLegendCyclic:
            legendDiameter = 6 * dx
            legendMargin = 2 * dx

            legendRect = QRect(0, 0, legendDiameter, legendDiameter)
            legendRect.moveRight(widgetRect.right() - legendMargin)
            legendRect.moveBottom(widgetRect.height() - legendMargin)

            cgradient = QConicalGradient(legendRect.center(), 90.)
            cgradient.setStops(self._colorLegendStopPoints)
            fgPainter.setBrush(cgradient)
            fgPainter.drawEllipse(legendRect)
        else:
            tickLabels = [f'{tick:5g}' for tick in self._colorLegendTicks]
            tickLabelWidth = max(fontMetrics.width(label) for label in tickLabels)

            legendWidth = 2 * dx
            legendHeight = (2 * len(tickLabels) - 1) * dy
            legendMargin = tickLabelWidth + 2 * dx

            legendRect = QRect(0, 0, legendWidth, legendHeight)
            legendRect.moveRight(widgetRect.right() - legendMargin)
            legendRect.moveTop((widgetRect.height() - legendHeight) // 2)

            lgradient = QLinearGradient(legendRect.bottomLeft(), legendRect.topLeft())
            lgradient.setStops(self._colorLegendStopPoints)
            fgPainter.setBrush(lgradient)
            fgPainter.drawRect(legendRect)

            tickX0 = legendRect.right() + dx
            tickY0 = legendRect.bottom() + fontMetrics.strikeOutPos()

            for tickIndex, tickLabel in enumerate(tickLabels):
                tickDY = (tickIndex * legendRect.height()) // (len(tickLabels) - 1)
                viewportPoint = QPoint(tickX0, tickY0 - tickDY)
                fgPainter.drawText(viewportPoint, tickLabel)

    def wheelEvent(self, event: QWheelEvent) -> None:
        oldPosition = self.mapToScene(event.pos())

        zoomBase = 1.25
        zoom = zoomBase if event.angleDelta().y() > 0 else 1. / zoomBase
        self.scale(zoom, zoom)

        newPosition = self.mapToScene(event.pos())

        deltaPosition = newPosition - oldPosition
        self.translate(deltaPosition.x(), deltaPosition.y())


class ImageView(QWidget):

    def __init__(self, statusBar: QStatusBar, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.imageRibbon = ImageRibbon.createInstance()
        self.imageWidget = ImageWidget.createInstance(statusBar)

    @classmethod
    def createInstance(cls, statusBar: QStatusBar, parent: Optional[QWidget] = None) -> ImageView:
        view = cls(statusBar, parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setMenuBar(view.imageRibbon)
        layout.addWidget(view.imageWidget)
        view.setLayout(layout)

        view.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

        return view
