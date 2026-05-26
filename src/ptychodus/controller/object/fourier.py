import logging

from PyQt5.QtCore import QRectF

from ptychodus.api.geometry import Box2D

from ...model.analysis import FourierAnalysisResult, FourierAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import FourierAnalysisDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController

logger = logging.getLogger(__name__)


class FourierAnalysisViewController:
    def __init__(
        self,
        analyzer: FourierAnalyzer,
        real_space_visualization_engine: VisualizationEngine,
        reciprocal_space_visualization_engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._product_index = -1
        self._result: FourierAnalysisResult | None = None

        self._dialog = FourierAnalysisDialog()
        self._dialog.setWindowTitle('Fourier Analysis')

        self._real_space_image_controller = ImageController(
            real_space_visualization_engine,
            self._dialog.real_space_view,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._reciprocal_space_image_controller = ImageController(
            reciprocal_space_visualization_engine,
            self._dialog.reciprocal_space_view,
            self._dialog.status_bar,
            file_dialog_factory,
        )

        item_signals = self._real_space_image_controller.get_item().get_signals()
        item_signals.fourier_finished.connect(self._analyze_fourier)

    def analyze(self, item_index: int) -> None:
        self._product_index = item_index

        try:
            self._result = self._analyzer.analyze_layer(item_index)
        except Exception as exc:
            self._result = None
            logger.exception(exc)
            ExceptionDialog.show_exception('Fourier Analysis', exc)

        self._sync_model_to_view()
        self._dialog.open()

    def _analyze_fourier(self, rect: QRectF) -> None:
        if rect.isEmpty():
            logger.debug('QRectF is empty!')
            return

        box = Box2D(
            x=rect.x(),
            y=rect.y(),
            width=rect.width(),
            height=rect.height(),
        )

        try:
            self._result = self._analyzer.analyze_roi(self._product_index, box)
        except ValueError as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Fourier Analysis', exc)
            return

        self._sync_model_to_view()

    def _sync_model_to_view(self) -> None:
        try:
            object_ = self._analyzer.get_object(self._product_index)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Fourier Analysis', exc)
        else:
            self._real_space_image_controller.set_array(
                object_.get_layer(0), object_.get_pixel_geometry()
            )

        if self._result is None:
            self._reciprocal_space_image_controller.clear_array()
        else:
            self._reciprocal_space_image_controller.set_array(
                self._result.transformed_roi, self._result.pixel_geometry
            )
