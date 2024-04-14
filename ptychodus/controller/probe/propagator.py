import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import ProbePropagator
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class ProbePropagationViewController:

    def __init__(self, propagator: ProbePropagator, engine: VisualizationEngine,
                 statusBar: QStatusBar, fileDialogFactory: FileDialogFactory,
                 parent: QWidget | None) -> None:
        super().__init__()
        self._propagator = propagator
        self._dialog = ProbePropagationDialog.createInstance(parent)
        self._xyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.xyView, statusBar, fileDialogFactory)
        self._zxVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zxView, statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.parametersView.visualizationParametersView)
        self._zyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zyView, statusBar, fileDialogFactory)

        # FIXME view.propagateButton = QPushButton('Propagate')
        # FIXME view.saveButton = QPushButton('Save')
        # FIXME view.coordinateSlider = QSlider(Qt.Orientation.Horizontal)
        # FIXME view.coordinateLabel = QLabel()

    def propagate(self, itemIndex: int) -> None:
        itemName = self._propagator.getName(itemIndex)
        _ = self._propagator.propagate(itemIndex)  # FIXME start, stop, count
        # FIXME do something with result
        self._dialog.setWindowTitle(f'Propagate Probe: {itemName}')
        self._dialog.open()
