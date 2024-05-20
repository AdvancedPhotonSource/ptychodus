from decimal import Decimal
import logging

from ...model.analysis import ProbePropagator, PropagatedProbe
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class ProbePropagationViewController:  # FIXME like Fluorescence

    def __init__(self, propagator: ProbePropagator, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._propagator = propagator
        self._fileDialogFactory = fileDialogFactory

        self._dialog = ProbePropagationDialog()
        self._dialog.parametersView.beginCoordinateWidget.setLengthInMeters(Decimal('-1e-3'))
        self._dialog.parametersView.endCoordinateWidget.setLengthInMeters(Decimal('+1e-3'))
        self._dialog.parametersView.numberOfStepsSpinBox.setRange(1, 999)
        self._dialog.propagateButton.clicked.connect(self._repropagate)
        self._dialog.saveButton.clicked.connect(self._saveResult)
        self._dialog.coordinateSlider.valueChanged.connect(self._updateCurrentCoordinate)

        self._xyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.xyView, self._dialog.statusBar, fileDialogFactory)
        self._zxVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zxView, self._dialog.statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.parametersView.visualizationParametersView)
        self._zyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zyView, self._dialog.statusBar, fileDialogFactory)
        self._result: PropagatedProbe | None = None

    def _updateCurrentCoordinate(self, step: int) -> None:
        result = self._result
        lerpValue = 0.

        if result is not None:
            slider = self._dialog.coordinateSlider
            upper = step - slider.minimum()
            lower = slider.maximum() - slider.minimum()

            if lower > 0:
                alpha = upper / lower
                lerpValue = (1 - alpha) * result.beginCoordinateInMeters \
                        + alpha * result.endCoordinateInMeters
            else:
                logger.error('Bad slider range!')

            self._xyVisualizationWidgetController.setArray(result.getXYProjection(step),
                                                           result.pixelGeometry)

        self._dialog.coordinateLabel.setText(f'{lerpValue:.4g}')

    def _propagate(self, itemIndex: int) -> None:
        parametersView = self._dialog.parametersView
        beginCoordinateInMeters = float(parametersView.beginCoordinateWidget.getLengthInMeters())
        endCoordinateInMeters = float(parametersView.endCoordinateWidget.getLengthInMeters())
        numberOfSteps = parametersView.numberOfStepsSpinBox.value()

        try:
            result = self._propagator.propagate(itemIndex, beginCoordinateInMeters,
                                                endCoordinateInMeters, numberOfSteps)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Propagator', err)
            self._result = None
            return

        self._result = result
        self._zxVisualizationWidgetController.setArray(result.getZXProjection(),
                                                       result.pixelGeometry)
        self._zyVisualizationWidgetController.setArray(result.getZYProjection(),
                                                       result.pixelGeometry)
        self._dialog.coordinateSlider.setRange(0, self._result.getNumberOfSteps() - 1)
        self._updateCurrentCoordinate(self._dialog.coordinateSlider.value())

    def _repropagate(self) -> None:
        if self._result is None:
            logger.debug('No result to repropagate!')
        else:
            self._propagate(self._result.itemIndex)

    def propagate(self, itemIndex: int) -> None:
        self._propagate(itemIndex)

        if self._result is None:
            logger.debug('No result to show!')
            return

        itemName = self._result.itemName
        self._dialog.setWindowTitle(f'Propagate Probe: {itemName}')
        self._dialog.open()

    def _saveResult(self) -> None:
        if self._result is None:
            logger.debug('No result to save!')
            return

        title = 'Save Propagated Probe'
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._dialog,
            title,
            nameFilters=self._propagator.getSaveFileFilterList(),
            selectedNameFilter=self._propagator.getSaveFileFilter())

        if filePath:
            try:
                self._propagator.savePropagatedProbe(filePath, self._result)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException(title, err)
