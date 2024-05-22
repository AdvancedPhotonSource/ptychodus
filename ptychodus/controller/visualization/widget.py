from __future__ import annotations

from PyQt5.QtWidgets import QStatusBar

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import NumberArrayType

from ...model.visualization import VisualizationEngine
from ...view.visualization import VisualizationWidget
from ..data import FileDialogFactory
from .controller import VisualizationController


class VisualizationWidgetController:

    def __init__(self, engine: VisualizationEngine, widget: VisualizationWidget,
                 statusBar: QStatusBar, fileDialogFactory: FileDialogFactory) -> None:
        self._widget = widget
        self._controller = VisualizationController.createInstance(engine, widget.visualizationView,
                                                                  statusBar, fileDialogFactory)

        self._widget.homeAction.triggered.connect(self._controller.zoomToFit)
        self._widget.saveAction.triggered.connect(self._controller.saveImage)
        self._widget.autoscaleAction.triggered.connect(self._autoDisplayRange)

    def _autoDisplayRange(self) -> None:
        self._controller.rerenderImage(autoscaleColorAxis=True)

    def setArray(self, array: NumberArrayType, pixelGeometry: PixelGeometry) -> None:
        self._controller.setArray(array, pixelGeometry)

    def clearArray(self) -> None:
        self._controller.clearArray()
