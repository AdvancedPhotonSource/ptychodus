from __future__ import annotations
from decimal import Decimal
import logging

import numpy

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QButtonGroup, QDialog, QStatusBar

from ptychodus.api.geometry import Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.typing import NumberArrayType

from ..model.visualization import VisualizationEngine
from ..view.image import (
    ImageDataRangeGroupBox,
    ImageDisplayRangeDialog,
    ImageRendererGroupBox,
    ImageToolsGroupBox,
    ImageView,
    ImageWidget,
)
from ..view.visualization import ImageItem, ImageMouseTool
from .data import FileDialogFactory
from .visualization import VisualizationController

logger = logging.getLogger(__name__)


class ImageToolsController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(
        self,
        view: ImageToolsGroupBox,
        visualization_controller: VisualizationController,
    ) -> None:
        self._view = view
        self._visualization_controller = visualization_controller

        view.home_button.clicked.connect(visualization_controller.zoom_to_fit)
        view.save_button.clicked.connect(visualization_controller.save_image)
        view.move_button.setCheckable(True)
        view.move_button.setChecked(True)
        view.fourier_button.setCheckable(True)
        view.ruler_button.setCheckable(True)
        view.rectangle_button.setCheckable(True)
        view.line_cut_button.setCheckable(True)

        self._mouse_tool_button_group = QButtonGroup()
        self._mouse_tool_button_group.addButton(view.move_button, ImageMouseTool.MOVE_TOOL.value)
        self._mouse_tool_button_group.addButton(
            view.fourier_button, ImageMouseTool.FOURIER_TOOL.value
        )
        self._mouse_tool_button_group.addButton(view.ruler_button, ImageMouseTool.RULER_TOOL.value)
        self._mouse_tool_button_group.addButton(
            view.rectangle_button, ImageMouseTool.RECTANGLE_TOOL.value
        )
        self._mouse_tool_button_group.addButton(
            view.line_cut_button, ImageMouseTool.LINE_CUT_TOOL.value
        )
        self._mouse_tool_button_group.idToggled.connect(self._set_mouse_tool)

    def _set_mouse_tool(self, tool_id: int, checked: bool) -> None:
        if checked:
            mouse_tool = ImageMouseTool(tool_id)
            self._visualization_controller.set_mouse_tool(mouse_tool)


class ImageRendererController(Observer):
    def __init__(self, engine: VisualizationEngine, view: ImageRendererGroupBox) -> None:
        super().__init__()
        self._engine = engine
        self._view = view

        self._renderer_model = QStringListModel()
        view.renderer_combo_box.setModel(self._renderer_model)

        self._transformation_model = QStringListModel()
        view.transformation_combo_box.setModel(self._transformation_model)

        self._variant_model = QStringListModel()
        view.variant_combo_box.setModel(self._variant_model)

        self._sync_model_to_view()
        engine.add_observer(self)

        view.renderer_combo_box.textActivated.connect(engine.set_renderer)
        view.transformation_combo_box.textActivated.connect(engine.set_transformation)
        view.variant_combo_box.textActivated.connect(engine.set_variant)

    def _sync_model_to_view(self) -> None:
        self._view.renderer_combo_box.blockSignals(True)
        self._renderer_model.setStringList([name for name in self._engine.renderers()])
        self._view.renderer_combo_box.setCurrentText(self._engine.get_renderer())
        self._view.renderer_combo_box.blockSignals(False)

        self._view.transformation_combo_box.blockSignals(True)
        self._transformation_model.setStringList([name for name in self._engine.transformations()])
        self._view.transformation_combo_box.setCurrentText(self._engine.get_transformation())
        self._view.transformation_combo_box.blockSignals(False)

        self._view.variant_combo_box.blockSignals(True)
        self._variant_model.setStringList([name for name in self._engine.variants()])
        self._view.variant_combo_box.setCurrentText(self._engine.get_variant())
        self._view.variant_combo_box.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._sync_model_to_view()


class ImageDataRangeController(Observer):
    def __init__(
        self,
        engine: VisualizationEngine,
        view: ImageDataRangeGroupBox,
        image_widget: ImageWidget,
        visualization_controller: VisualizationController,
    ) -> None:
        self._engine = engine
        self._view = view
        self._image_widget = image_widget
        self._display_range_dialog = ImageDisplayRangeDialog(view)
        self._visualization_controller = visualization_controller
        self._display_range_is_locked = True

        self._sync_model_to_view()
        engine.add_observer(self)

        view.min_display_value_slider.value_changed.connect(
            lambda value: engine.set_min_display_value(float(value))
        )
        view.max_display_value_slider.value_changed.connect(
            lambda value: engine.set_max_display_value(float(value))
        )
        view.auto_button.clicked.connect(self._auto_display_range)
        view.edit_button.clicked.connect(self._display_range_dialog.open)
        self._display_range_dialog.finished.connect(self._finish_editing_display_range)

        view.color_legend_button.setCheckable(True)
        image_widget.set_color_legend_visible(view.color_legend_button.isChecked())
        view.color_legend_button.toggled.connect(image_widget.set_color_legend_visible)

    def _auto_display_range(self) -> None:
        self._display_range_is_locked = False
        self._visualization_controller.rerender_image(autoscale_color_axis=True)
        self._display_range_is_locked = True

    def _finish_editing_display_range(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            lower = float(self._display_range_dialog.min_value_line_edit.get_value())
            upper = float(self._display_range_dialog.max_value_line_edit.get_value())

            self._display_range_is_locked = False
            self._engine.set_display_value_range(lower, upper)
            self._display_range_is_locked = True

    def _sync_color_legend_to_view(self) -> None:
        values = numpy.linspace(
            self._engine.get_min_display_value(), self._engine.get_max_display_value(), 1000
        )
        self._image_widget.set_color_legend_colors(
            values,
            self._engine.colorize(values),
            self._engine.is_renderer_cyclic(),
        )

    def _sync_model_to_view(self) -> None:
        min_value = Decimal(repr(self._engine.get_min_display_value()))
        max_value = Decimal(repr(self._engine.get_max_display_value()))

        self._display_range_dialog.min_value_line_edit.set_value(min_value)
        self._display_range_dialog.max_value_line_edit.set_value(max_value)

        if self._display_range_is_locked:
            self._view.min_display_value_slider.set_value(min_value)
            self._view.max_display_value_slider.set_value(max_value)
        else:
            display_range_limits = Interval[Decimal](min_value, max_value)
            self._view.min_display_value_slider.set_value_and_range(min_value, display_range_limits)
            self._view.max_display_value_slider.set_value_and_range(max_value, display_range_limits)

        self._sync_color_legend_to_view()

    def _update(self, observable: Observable) -> None:
        if observable is self._engine:
            self._sync_model_to_view()


class ImageController:
    def __init__(
        self,
        engine: VisualizationEngine,
        view: ImageView,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._visualization_controller = VisualizationController(
            engine, view.image_widget, status_bar, file_dialog_factory
        )
        self._tools_controller = ImageToolsController(
            view.image_ribbon.image_tools_group_box, self._visualization_controller
        )
        self._renderer_controller = ImageRendererController(
            engine, view.image_ribbon.colormap_group_box
        )
        self._data_range_controller = ImageDataRangeController(
            engine,
            view.image_ribbon.data_range_group_box,
            view.image_widget,
            self._visualization_controller,
        )

    def get_item(self) -> ImageItem:
        return self._visualization_controller.get_item()

    def set_array(self, array: NumberArrayType, pixel_geometry: PixelGeometry) -> None:
        self._visualization_controller.set_array(array, pixel_geometry)

    def clear_array(self) -> None:
        self._visualization_controller.clear_array()
