import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import FourierAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import FourierAnalysisDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationWidgetController

logger = logging.getLogger(__name__)


class FourierAnalysisViewController(Observer):
    def __init__(
        self,
        analyzer: FourierAnalyzer,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._engine = engine

        self._dialog = FourierAnalysisDialog()
        self._dialog.setWindowTitle('Fourier Analysis')

        self._real_space_widget_controller = VisualizationWidgetController(
            engine, self._dialog.real_space_widget, self._dialog.status_bar, FileDialogFactory()
        )
        self._reciprocal_space_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.reciprocal_space_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )

        # FIXME call from vis tool: self._analyzer.analyze_roi(bounding_box)

        analyzer.add_observer(self)

    def analyze(self, item_index: int) -> None:
        self._analyzer.set_product(item_index)
        self._dialog.open()

    def _sync_model_to_view(self) -> None:
        try:
            object_ = self._analyzer.get_object()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Fourier Analysis', err)
        else:
            self._real_space_widget_controller.set_array(
                object_.get_layer(0), object_.get_pixel_geometry()
            )

        try:
            result = self._analyzer.get_result()
        except ValueError:
            self._reciprocal_space_widget_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Fourier Analysis', err)
        else:
            self._reciprocal_space_widget_controller.set_array(
                result.transformed_roi, result.pixel_geometry
            )

    def _update(self, observable: Observable) -> None:
        if observable is self._analyzer:
            self._sync_model_to_view()
