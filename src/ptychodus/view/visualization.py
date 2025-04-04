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
    rectangle_finished = pyqtSignal(QRectF)
    line_cut_finished = pyqtSignal(QLineF)


class ImageMouseTool(Enum):
    MOVE_TOOL = auto()
    RULER_TOOL = auto()
    RECTANGLE_TOOL = auto()
    LINE_CUT_TOOL = auto()


class ImageItem(QGraphicsPixmapItem):
    def __init__(self, events: ImageItemEvents, status_bar: QStatusBar) -> None:
        super().__init__()
        self._events = events
        self._status_bar = status_bar
        self._product: VisualizationProduct | None = None
        self._mouse_tool = ImageMouseTool.MOVE_TOOL
        self._line_item = QGraphicsLineItem(self)
        self._line_item.hide()
        self._rectangle_item = QGraphicsRectItem(self)
        self._rectangle_item.hide()
        self._rectangle_origin = QPointF()
        self.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)

    def get_product(self) -> VisualizationProduct | None:
        return self._product

    def set_product(self, product: VisualizationProduct) -> None:
        image_rgba_f = product.get_image_rgba()
        # NOTE .copy() ensures imageRGBAi is not a view
        image_rgba_i = numpy.multiply(image_rgba_f, 255).astype(numpy.uint8).copy()

        try:
            image = QImage(
                image_rgba_i.data,
                image_rgba_i.shape[1],
                image_rgba_i.shape[0],
                image_rgba_i.strides[0],
                QImage.Format.Format_RGBA8888,
            )
            pixmap = QPixmap.fromImage(image)
        except Exception as exc:
            logger.exception(exc)
            pixmap = QPixmap()

        self._product = product
        self.setPixmap(pixmap)

    def clear_product(self) -> None:
        self._product = None
        self.setPixmap(QPixmap())

    def set_mouse_tool(self, mouse_tool: ImageMouseTool) -> None:
        self._mouse_tool = mouse_tool

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        app = QApplication.instance()

        if app:
            cursor = Qt.CursorShape.CrossCursor

            if self._mouse_tool == ImageMouseTool.MOVE_TOOL:
                cursor = Qt.CursorShape.OpenHandCursor

            app.setOverrideCursor(cursor)  # type: ignore

        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        pos = event.pos()

        if self._product is not None:
            info_text = self._product.get_info_text(pos.x(), pos.y())
            self._status_bar.showMessage(info_text)

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:  # noqa: N802
        app = QApplication.instance()

        if app:
            app.restoreOverrideCursor()  # type: ignore

        self._status_bar.clearMessage()
        super().hoverLeaveEvent(event)

    def _change_override_cursor(self, cursor: Qt.CursorShape) -> None:
        app = QApplication.instance()

        if app:
            app.changeOverrideCursor(cursor)  # type: ignore

    @staticmethod
    def _create_pen(color: Qt.GlobalColor) -> QPen:
        pen = QPen(color)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        pen.setCosmetic(True)
        return pen

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._mouse_tool == ImageMouseTool.MOVE_TOOL:
            self._change_override_cursor(Qt.CursorShape.ClosedHandCursor)
        elif self._mouse_tool == ImageMouseTool.RULER_TOOL:
            line = QLineF(event.pos(), event.pos())
            self.prepareGeometryChange()
            self._line_item.setLine(line)
            self._line_item.setPen(self._create_pen(Qt.GlobalColor.cyan))
            self._line_item.show()
        elif self._mouse_tool == ImageMouseTool.RECTANGLE_TOOL:
            self._rectangle_origin = event.pos()
            rect = QRectF(self._rectangle_origin, QSizeF())
            self.prepareGeometryChange()
            self._rectangle_item.setRect(rect)
            self._rectangle_item.setPen(self._create_pen(Qt.GlobalColor.cyan))
            self._rectangle_item.show()
        elif self._mouse_tool == ImageMouseTool.LINE_CUT_TOOL:
            line = QLineF(event.pos(), event.pos())
            self.prepareGeometryChange()
            self._line_item.setLine(line)
            self._line_item.setPen(self._create_pen(Qt.GlobalColor.magenta))
            self._line_item.show()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._mouse_tool == ImageMouseTool.MOVE_TOOL:
            self.setPos(self.scenePos() + event.scenePos() - event.lastScenePos())
        elif self._mouse_tool == ImageMouseTool.RULER_TOOL:
            origin = self._line_item.line().p1()
            line = QLineF(origin, event.pos())
            self.prepareGeometryChange()
            self._line_item.setLine(line)
            message1 = f'{line.length():.1f} pixels, {line.angle():.2f}\u00b0'
            message2 = f'{line.dx():.1f} \u00d7 {line.dy():.1f}'
            self._status_bar.showMessage(f'{message1} ({message2})')
        elif self._mouse_tool == ImageMouseTool.RECTANGLE_TOOL:
            rect = QRectF(self._rectangle_origin, event.pos()).normalized()
            center = rect.center()
            self.prepareGeometryChange()
            self._rectangle_item.setRect(rect)
            message1 = f'{rect.width():.1f} \u00d7 {rect.height():.1f}'
            message2 = f'{center.x():.1f}, {center.y():.1f}'
            self._status_bar.showMessage(f'Rectangle: {message1} (Center: {message2})')
        elif self._mouse_tool == ImageMouseTool.LINE_CUT_TOOL:
            origin = self._line_item.line().p1()
            line = QLineF(origin, event.pos())
            self.prepareGeometryChange()
            self._line_item.setLine(line)
            message1 = f'{line.length():.1f} pixels, {line.angle():.2f}\u00b0'
            message2 = f'{line.dx():.1f} \u00d7 {line.dy():.1f}'
            self._status_bar.showMessage(f'{message1} ({message2})')

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        if self._mouse_tool == ImageMouseTool.MOVE_TOOL:
            self._change_override_cursor(Qt.CursorShape.OpenHandCursor)
        elif self._mouse_tool == ImageMouseTool.RULER_TOOL:
            self._line_item.setLine(QLineF())
            self._line_item.hide()
        elif self._mouse_tool == ImageMouseTool.RECTANGLE_TOOL:
            self._events.rectangle_finished.emit(self._rectangle_item.rect())
            self._rectangle_item.setRect(QRectF())
            self._rectangle_item.hide()
        elif self._mouse_tool == ImageMouseTool.LINE_CUT_TOOL:
            self._events.line_cut_finished.emit(self._line_item.line())
            self._line_item.setLine(QLineF())
            self._line_item.hide()


class LineCutDialog(QDialog):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.axes = self.figure.add_subplot(111)

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> LineCutDialog:
        view = cls(parent)
        view.setWindowTitle('Line-Cut Dialog')

        layout = QVBoxLayout()
        layout.addWidget(view.navigation_toolbar)
        layout.addWidget(view.figure_canvas)
        view.setLayout(layout)

        return view


class RectangleView(QGroupBox):
    @staticmethod
    def _create_read_only_line_edit() -> QLineEdit:
        line_edit = QLineEdit()

        palette = line_edit.palette()
        palette.setColor(QPalette.Base, palette.color(QPalette.Window))
        palette.setColor(QPalette.Text, palette.color(QPalette.WindowText))
        line_edit.setPalette(palette)

        line_edit.setFocusPolicy(Qt.NoFocus)
        line_edit.setReadOnly(True)

        return line_edit

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Rectangle', parent)
        self.center_x_line_edit = RectangleView._create_read_only_line_edit()
        self.center_y_line_edit = RectangleView._create_read_only_line_edit()
        self.width_line_edit = RectangleView._create_read_only_line_edit()
        self.height_line_edit = RectangleView._create_read_only_line_edit()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> RectangleView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Center X:', view.center_x_line_edit)
        layout.addRow('Center Y:', view.center_y_line_edit)
        layout.addRow('Width:', view.width_line_edit)
        layout.addRow('Height:', view.height_line_edit)
        view.setLayout(layout)

        return view


class HistogramDialog(QDialog):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.figure = Figure()
        self.figure_canvas = FigureCanvasQTAgg(self.figure)
        self.navigation_toolbar = NavigationToolbar(self.figure_canvas, self)
        self.axes = self.figure.add_subplot(111)
        self.rectangle_view = RectangleView.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> HistogramDialog:
        view = cls(parent)
        view.setWindowTitle('Histogram')

        layout = QVBoxLayout()
        layout.addWidget(view.navigation_toolbar)
        layout.addWidget(view.figure_canvas, 1)
        layout.addWidget(view.rectangle_view)
        view.setLayout(layout)

        return view


class VisualizationView(QGraphicsView):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        old_position = self.mapToScene(event.pos())

        zoom_base = 1.25
        zoom = zoom_base if event.angleDelta().y() > 0 else 1.0 / zoom_base
        self.scale(zoom, zoom)

        new_position = self.mapToScene(event.pos())

        delta_position = new_position - old_position
        self.translate(delta_position.x(), delta_position.y())


class VisualizationWidget(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None) -> None:
        super().__init__(title, parent)
        self.tool_bar = QToolBar('Tools')
        self.home_action = QAction(QIcon(':/icons/home'), 'Home')
        self.save_action = QAction(QIcon(':/icons/save'), 'Save Image')
        self.autoscale_action = QAction(QIcon(':/icons/autoscale'), 'Autoscale Color Axis')
        self.visualization_view = VisualizationView()

    @classmethod
    def create_instance(cls, title: str, parent: QWidget | None = None) -> VisualizationWidget:
        view = cls(title, parent)
        view.setAlignment(Qt.AlignHCenter)

        view.tool_bar.setFloatable(False)
        view.tool_bar.setIconSize(QSize(32, 32))
        view.tool_bar.addAction(view.home_action)
        view.tool_bar.addAction(view.save_action)
        view.tool_bar.addAction(view.autoscale_action)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.tool_bar)
        layout.addWidget(view.visualization_view)
        view.setLayout(layout)

        return view


class VisualizationParametersView(QGroupBox):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Visualization', parent)

        self.renderer_combo_box = QComboBox()
        self.transformation_combo_box = QComboBox()
        self.variant_combo_box = QComboBox()
        self.min_display_value_line_edit = DecimalLineEdit.create_instance()
        self.max_display_value_line_edit = DecimalLineEdit.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> VisualizationParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Renderer:', view.renderer_combo_box)
        layout.addRow('Transform:', view.transformation_combo_box)
        layout.addRow('Variant:', view.variant_combo_box)
        layout.addRow('Min Display Value:', view.min_display_value_line_edit)
        layout.addRow('Max Display Value:', view.max_display_value_line_edit)
        view.setLayout(layout)

        return view
