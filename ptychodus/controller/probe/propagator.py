import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import ProbePropagator, PropagatedProbe
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ...view.widgets import ExceptionDialog
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
        self._result: PropagatedProbe | None = None

        # FIXME init self._dialog.parametersView
        # FIXME view.propagateButton = QPushButton('Propagate')
        # FIXME view.saveButton = QPushButton('Save')
        # FIXME view.coordinateSlider = QSlider(Qt.Orientation.Horizontal)
        # FIXME view.coordinateLabel = QLabel()

    def propagate(self, itemIndex: int) -> None:
        parametersView = self._dialog.parametersView
        beginCoordinateInMeters = parametersView.beginCoordinateWidget.getLengthInMeters()
        endCoordinateInMeters = parametersView.endCoordinateWidget.getLengthInMeters()
        numberOfSteps = parametersView.numberOfStepsSpinBox.value()

        try:
            result = self._propagator.propagate(itemIndex, float(beginCoordinateInMeters),
                                                float(endCoordinateInMeters), numberOfSteps)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Propagator', err)
            return

        self._result = result
        # FIXME result to view
        self._dialog.setWindowTitle(f'Propagate Probe: {result.itemName}')
        self._dialog.open()
