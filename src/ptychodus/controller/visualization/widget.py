from __future__ import annotations

from PyQt5.QtWidgets import QAction, QActionGroup, QStatusBar

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.typing import NumberArrayType

from ...model.visualization import VisualizationEngine
from ...view.visualization import ImageItem, ImageMouseTool, VisualizationWidget
from ..data import FileDialogFactory
from .controller import VisualizationController


class VisualizationWidgetController:
    def __init__(
        self,
        engine: VisualizationEngine,
        widget: VisualizationWidget,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._widget = widget
        self._controller = VisualizationController(
            engine, widget.visualization_view, status_bar, file_dialog_factory
        )

        widget.home_action.triggered.connect(self._controller.zoom_to_fit)
        widget.save_action.triggered.connect(self._controller.save_image)
        widget.autoscale_action.triggered.connect(self._auto_display_range)

        widget.move_action.setData(ImageMouseTool.MOVE_TOOL.value)
        widget.move_action.setCheckable(True)
        widget.move_action.setChecked(True)

        widget.ruler_action.setData(ImageMouseTool.RULER_TOOL.value)
        widget.ruler_action.setCheckable(True)

        widget.rectangle_action.setData(ImageMouseTool.RECTANGLE_TOOL.value)
        widget.rectangle_action.setCheckable(True)

        widget.line_cut_action.setData(ImageMouseTool.LINE_CUT_TOOL.value)
        widget.line_cut_action.setCheckable(True)

        self._mouse_tool_action_group = QActionGroup(widget.tool_bar)
        self._mouse_tool_action_group.addAction(widget.move_action)
        self._mouse_tool_action_group.addAction(widget.ruler_action)
        self._mouse_tool_action_group.addAction(widget.rectangle_action)
        self._mouse_tool_action_group.addAction(widget.line_cut_action)
        self._mouse_tool_action_group.triggered.connect(self._set_mouse_tool)

    def get_item(self) -> ImageItem:
        return self._controller.get_item()

    def _auto_display_range(self) -> None:
        self._controller.rerender_image(autoscale_color_axis=True)

    def set_array(self, array: NumberArrayType, pixel_geometry: PixelGeometry) -> None:
        self._controller.set_array(array, pixel_geometry)

    def clear_array(self) -> None:
        self._controller.clear_array()

    def _set_mouse_tool(self, mouse_tool_action: QAction) -> None:
        tool_id = mouse_tool_action.data()
        mouse_tool = ImageMouseTool(tool_id)
        self._controller.set_mouse_tool(mouse_tool)
