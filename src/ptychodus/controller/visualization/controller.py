from __future__ import annotations
import logging

import numpy

from PyQt5.QtCore import Qt, QLineF, QRectF
from PyQt5.QtWidgets import QGraphicsScene, QStatusBar

from ptychodus.api.geometry import Box2D, Line2D, PixelGeometry, Point2D
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.typing import NumberArrayType

from ...model.visualization import VisualizationEngine
from ...view.visualization import (
    HistogramDialog,
    ImageItem,
    ImageItemSignals,
    ImageMouseTool,
    LineCutDialog,
    VisualizationView,
)
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory

logger = logging.getLogger(__name__)


class VisualizationController(Observer):
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(
        self,
        engine: VisualizationEngine,
        view: VisualizationView,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._view = view
        self._status_bar = status_bar
        self._file_dialog_factory = file_dialog_factory
        self._line_cut_dialog = LineCutDialog(view)
        self._histogram_dialog = HistogramDialog(view)

        item_signals = ImageItemSignals()
        item_signals.line_cut_finished.connect(self._analyze_line_cut)
        item_signals.rectangle_finished.connect(self._analyze_region)

        self._item = ImageItem(item_signals, status_bar)
        engine.add_observer(self)

        scene = QGraphicsScene()
        scene.addItem(self._item)
        view.setScene(scene)

        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def get_item(self) -> ImageItem:
        return self._item

    def set_array(
        self,
        array: NumberArrayType,
        pixel_geometry: PixelGeometry,
        *,
        autoscale_color_axis: bool = False,
    ) -> None:
        if numpy.all(numpy.isfinite(array)):
            try:
                product = self._engine.render(
                    array, pixel_geometry, autoscale_color_axis=autoscale_color_axis
                )
            except ValueError as err:
                logger.exception(err)
                ExceptionDialog.show_exception('Renderer', err)
            else:
                self._item.set_product(product)
        else:
            logger.warning('Array contains infinite or NaN values!')
            self._item.clear_product()

    def clear_array(self) -> None:
        self._item.clear_product()

    def set_mouse_tool(self, mouse_tool: ImageMouseTool) -> None:
        self._item.set_mouse_tool(mouse_tool)

    def save_image(self) -> None:
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._view, 'Save Image', mime_type_filters=VisualizationController.MIME_TYPES
        )

        if file_path:
            pixmap = self._item.pixmap()
            pixmap.save(str(file_path))

    def _analyze_line_cut(self, line: QLineF) -> None:
        p1 = Point2D(line.x1(), line.y1())
        p2 = Point2D(line.x2(), line.y2())
        line2d = Line2D(p1, p2)

        product = self._item.get_product()

        if product is None:
            logger.warning('No visualization product!')
            return

        value_label = product.get_value_label()
        line_cut = product.get_line_cut(line2d)

        ax = self._line_cut_dialog.axes
        ax.clear()
        ax.plot(line_cut.distance_m, line_cut.value, '.-', linewidth=1.5)
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel(value_label)
        ax.grid(True)
        self._line_cut_dialog.figure_canvas.draw()
        self._line_cut_dialog.open()

    def _analyze_region(self, rect: QRectF) -> None:
        if rect.isEmpty():
            logger.debug('QRectF is empty!')
            return

        box = Box2D(
            x=rect.x(),
            y=rect.y(),
            width=rect.width(),
            height=rect.height(),
        )

        product = self._item.get_product()

        if product is None:
            logger.warning('No visualization product!')
            return

        value_label = product.get_value_label()
        kde = product.estimate_kernel_density(box)
        values = numpy.linspace(kde.value_lower, kde.value_upper, 1000)

        ax = self._histogram_dialog.axes
        ax.clear()
        ax.plot(values, kde.kde(values), '.-', linewidth=1.5)
        ax.set_xlabel(value_label)
        ax.set_ylabel('Density')
        ax.grid(True)
        self._histogram_dialog.figure_canvas.draw()

        rectangle_view = self._histogram_dialog.rectangle_view
        rect_center = rect.center()
        rectangle_view.center_x_line_edit.setText(f'{rect_center.x():.1f}')
        rectangle_view.center_y_line_edit.setText(f'{rect_center.y():.1f}')
        rectangle_view.width_line_edit.setText(f'{rect.width():.1f}')
        rectangle_view.height_line_edit.setText(f'{rect.height():.1f}')

        # TODO use rect for crop
        self._histogram_dialog.open()

    def zoom_to_fit(self) -> None:
        self._item.setPos(0, 0)
        scene = self._view.scene()

        if scene is None:
            raise ValueError('scene is None!')

        bounding_rect = scene.itemsBoundingRect()
        scene.setSceneRect(bounding_rect)
        self._view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def rerender_image(self, *, autoscale_color_axis: bool = False) -> None:
        product = self._item.get_product()

        if product is not None:
            self.set_array(
                product.get_values(),
                product.get_pixel_geometry(),
                autoscale_color_axis=autoscale_color_axis,
            )

    def _update(self, observable: Observable) -> None:
        if observable is self._engine:
            self.rerender_image()
