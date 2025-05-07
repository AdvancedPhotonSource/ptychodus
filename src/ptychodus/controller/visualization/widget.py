from __future__ import annotations

from PyQt5.QtWidgets import QStatusBar

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.typing import NumberArrayType

from ...model.visualization import VisualizationEngine
from ...view.visualization import VisualizationWidget
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
        self._controller = VisualizationController.create_instance(
            engine, widget.visualization_view, status_bar, file_dialog_factory
        )

        self._widget.home_action.triggered.connect(self._controller.zoom_to_fit)
        self._widget.save_action.triggered.connect(self._controller.save_image)
        self._widget.autoscale_action.triggered.connect(self._auto_display_range)

    def _auto_display_range(self) -> None:
        self._controller.rerender_image(autoscale_color_axis=True)

    def set_array(self, array: NumberArrayType, pixel_geometry: PixelGeometry) -> None:
        self._controller.set_array(array, pixel_geometry)

    def clear_array(self) -> None:
        self._controller.clear_array()
