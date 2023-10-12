from __future__ import annotations
from decimal import Decimal
from typing import Optional

from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, QSize
from PyQt5.QtGui import QIcon, QLinearGradient, QPainter, QPen, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGraphicsPixmapItem, QGraphicsScene, QGraphicsSceneHoverEvent,
                             QGraphicsSceneMouseEvent, QGraphicsView, QGridLayout, QHBoxLayout,
                             QPushButton, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
                             QStatusBar)

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
        self.moveButton = QToolButton()  # FIXME
        self.measureButton = QToolButton()  # FIXME

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ImageToolsGroupBox:
        view = cls(parent)

        view.homeButton.setIcon(QIcon(':/icons/home'))
        view.homeButton.setIconSize(QSize(64, 64))
        view.homeButton.setToolTip('Home')

        view.saveButton.setIcon(QIcon(':/icons/save'))
        view.saveButton.setIconSize(QSize(32, 32))
        view.saveButton.setToolTip('Save Image')

        view.moveButton.setIcon(QIcon(':/icons/move'))
        view.moveButton.setIconSize(QSize(32, 32))
        view.moveButton.setToolTip('Move')

        view.measureButton.setIcon(QIcon(':/icons/measure'))
        view.measureButton.setIconSize(QSize(32, 32))
        view.measureButton.setToolTip('Measure')

        layout = QGridLayout()
        layout.addWidget(view.homeButton, 0, 0, 1, 3)
        layout.setAlignment(view.homeButton, Qt.AlignHCenter)
        layout.addWidget(view.saveButton, 1, 0)
        layout.addWidget(view.moveButton, 1, 1)
        layout.addWidget(view.measureButton, 1, 2)
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


class ImageItem(QGraphicsPixmapItem):

    def __init__(self, statusBar: QStatusBar) -> None:
        super().__init__()
        self._statusBar = statusBar
        self._useMoveTool = True
        self.setTransformationMode(Qt.FastTransformation)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        app = QApplication.instance()

        if app:
            cursor = Qt.OpenHandCursor if self._useMoveTool else Qt.CrossCursor
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

        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._useMoveTool:
            app = QApplication.instance()

            if app:
                app.changeOverrideCursor(Qt.ClosedHandCursor)  # type: ignore

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._useMoveTool:
            self.setPos(self.scenePos() + event.scenePos() - event.lastScenePos())

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            pass
        elif event.button() == Qt.RightButton:
            self._useMoveTool = not self._useMoveTool
        else:
            return

        app = QApplication.instance()

        if app:
            cursor = Qt.OpenHandCursor if self._useMoveTool else Qt.CrossCursor
            app.changeOverrideCursor(cursor)  # type: ignore


class ImageWidget(QGraphicsView):

    def __init__(self, imageItem: ImageItem, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self._imageItem = imageItem
        self._isColorLegendVisible = False

    @classmethod
    def createInstance(cls,
                       statusBar: QStatusBar,
                       parent: Optional[QWidget] = None) -> ImageWidget:
        imageItem = ImageItem(statusBar)
        widget = cls(imageItem, parent)

        scene = QGraphicsScene()
        scene.addItem(imageItem)
        widget.setScene(scene)

        return widget

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._imageItem.setPixmap(pixmap)

    def getPixmap(self) -> QPixmap:
        return self._imageItem.pixmap()

    def setColorLegendVisible(self, visible: bool):
        self._isColorLegendVisible = visible
        self.scene().update()  # forces redraw

    def zoomToFit(self) -> None:
        self._imageItem.setPos(0, 0)
        scene = self.scene()
        boundingRect = scene.itemsBoundingRect()
        scene.setSceneRect(boundingRect)
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        # Draws the foreground of the scene using painter, after the background
        # and all items are drawn. All painting is done in scene coordinates.
        # rect is the exposed rectangle.
        if self._isColorLegendVisible:
            painter.save()

            # cosmetic pens have the same thickness at different scale factors
            pen = QPen()
            pen.setCosmetic(True)
            pen.setWidth(3)
            painter.setPen(pen)

            fontMetrics = painter.fontMetrics()
            dx = fontMetrics.horizontalAdvance('m')
            dy = fontMetrics.lineSpacing()

            tickLabels = ['1.00' + str(idx + 1) for idx in range(8)]  # FIXME tick labels
            tickLabelWidth = max(fontMetrics.horizontalAdvance(label) for label in tickLabels)

            legendWidth = 2 * dx
            legendHeight = (2 * len(tickLabels) - 1) * dy
            legendHMargin = tickLabelWidth + 2 * dx

            legendViewportRect = QRect(0, 0, legendWidth, legendHeight)
            widgetViewportRect = self.viewport().rect()
            legendViewportRect.moveRight(widgetViewportRect.right() - legendHMargin)
            legendViewportRect.moveTop((widgetViewportRect.height() - legendHeight) // 2)
            tickX0 = legendViewportRect.right() + dx
            tickY0 = legendViewportRect.bottom() + fontMetrics.strikeOutPos()

            legendScenePoly = self.mapToScene(legendViewportRect)
            legendSceneRect = legendScenePoly.boundingRect()
            gradient = QLinearGradient(legendSceneRect.bottomLeft(), legendSceneRect.topLeft())
            gradient.setColorAt(0.0, Qt.green)  # FIXME gradient value
            gradient.setColorAt(0.5, Qt.yellow)  # FIXME gradient value
            gradient.setColorAt(1.0, Qt.red)  # FIXME gradient value
            painter.setBrush(gradient)
            painter.drawPolygon(legendScenePoly)

            for tickIndex, tickLabel in enumerate(tickLabels):
                tickDY = (tickIndex * legendViewportRect.height()) // (len(tickLabels) - 1)
                viewportPoint = QPoint(tickX0, tickY0 - tickDY)
                scenePoint = self.mapToScene(viewportPoint)
                painter.drawText(scenePoint, tickLabel)

            painter.restore()

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
