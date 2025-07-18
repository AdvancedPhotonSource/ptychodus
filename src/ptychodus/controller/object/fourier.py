import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import FourierAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import FourierAnalysisDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationWidgetController

logger = logging.getLogger(__name__)


class FourierAnalysisViewController(Observer):
    def __init__(
        self,
        analyzer: FourierAnalyzer,
        engine: VisualizationEngine,
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
            FileDialogFactory(),
        )

        analyzer.add_observer(self)

    def analyze(self, item_index: int) -> None:
        # Perform Fourier analysis on the item at the given index
        self._analyzer.set_product(item_index)
        self._analyzer.analyze()
        self._dialog.open()

    def _sync_model_to_view(self) -> None:
        pass  # FIXME

    def _update(self, observable: Observable) -> None:
        if observable is self._analyzer:
            self._sync_model_to_view()
