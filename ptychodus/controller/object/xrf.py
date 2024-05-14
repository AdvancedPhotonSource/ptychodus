import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import FluorescenceEnhancer
from ...model.visualization import VisualizationEngine
from ...view.object import FluorescenceDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class FluorescenceViewController:

    def __init__(self, enhancer: FluorescenceEnhancer, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory, statusBar: QStatusBar,
                 parent: QWidget | None) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._engine = engine
        self._fileDialogFactory = fileDialogFactory
        self._statusBar = statusBar
        self._dialog = FluorescenceDialog.createInstance(parent)

        self._measuredWidgetController = VisualizationWidgetController(
            engine, self._dialog.measuredWidget, statusBar, fileDialogFactory)
        self._enhancedWidgetController = VisualizationWidgetController(
            engine, self._dialog.enhancedWidget, statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.visualizationParametersView)

    def enhanceXRF(self, itemIndex: int) -> None:
        print(itemIndex)  # FIXME
        self._dialog.open()
