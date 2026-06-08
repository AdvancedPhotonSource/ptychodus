from __future__ import annotations
import logging

from PyQt5.QtWidgets import QStatusBar

from ...model.analysis import ResidualAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.product import ProductVisualizationView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .core import ProductController

logger = logging.getLogger(__name__)


class ProductVisualizationController:
    def __init__(
        self,
        analyzer: ResidualAnalyzer,
        real_space_engine: VisualizationEngine,
        reciprocal_space_engine: VisualizationEngine,
        product_controller: ProductController,
        view: ProductVisualizationView,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._analyzer = analyzer
        self._product_controller = product_controller
        self._view = view

        self._real_image_controller = ImageController(
            real_space_engine,
            view.real_space_image_view,
            status_bar,
            file_dialog_factory,
        )
        self._reciprocal_image_controller = ImageController(
            reciprocal_space_engine,
            view.reciprocal_space_image_view,
            status_bar,
            file_dialog_factory,
        )

        view.compute_button.clicked.connect(self._compute)

    def _compute(self) -> None:
        product_index = self._product_controller.get_current_item_index()

        if product_index < 0:
            logger.warning('No current product!')
            return

        try:
            result = self._analyzer.analyze(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Residual Analyzer', err)
            return

        self._real_image_controller.set_array(
            result.real_space_error_map, result.object_pixel_geometry
        )
        self._reciprocal_image_controller.set_array(
            result.reciprocal_space_error_map, result.detector_pixel_geometry
        )
