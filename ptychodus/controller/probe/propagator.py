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
        self._fileDialogFactory = fileDialogFactory

        self._dialog = ProbePropagationDialog.createInstance(parent)
        self._dialog.propagateButton.clicked.connect(self._repropagate)
        self._dialog.saveButton.clicked.connect(self._saveResult)
        self._dialog.coordinateSlider.valueChanged.connect(self._updateCoordinateLabel)

        self._xyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.xyView, statusBar, fileDialogFactory)
        self._zxVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zxView, statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.parametersView.visualizationParametersView)
        self._zyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zyView, statusBar, fileDialogFactory)
        self._result: PropagatedProbe | None = None

    def _updateCoordinateLabel(self, value: int) -> None:
        lerpValue = 0.

        if self._result is not None:
            slider = self._dialog.coordinateSlider
            upper = value - slider.minimum()
            lower = slider.maximum() - slider.minimum()
            alpha = upper / lower

            beginValue = self._result.beginCoordinateInMeters
            endValue = self._result.endCoordinateInMeters
            lerpValue = (1 - alpha) * beginValue + alpha * endValue

        self._dialog.coordinateLabel.setText(f'{lerpValue:.4g}')

    def _propagate(self, itemIndex: int) -> None:
        parametersView = self._dialog.parametersView
        beginCoordinateInMeters = parametersView.beginCoordinateWidget.getLengthInMeters()
        endCoordinateInMeters = parametersView.endCoordinateWidget.getLengthInMeters()
        numberOfSteps = parametersView.numberOfStepsSpinBox.value()
        step = self._dialog.coordinateSlider.value()  # FIXME verify

        try:
            result = self._propagator.propagate(itemIndex, float(beginCoordinateInMeters),
                                                float(endCoordinateInMeters), numberOfSteps)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Propagator', err)
            return

        self._result = result
        self._xyVisualizationWidgetController.setArray(result.getXYProjection(step),
                                                       result.pixelGeometry)
        self._zxVisualizationWidgetController.setArray(result.getZXProjection(),
                                                       result.pixelGeometry)
        self._zyVisualizationWidgetController.setArray(result.getZYProjection(),
                                                       result.pixelGeometry)
        self._dialog.coordinateSlider.setRange(0, self._result.getNumberOfSteps())

    def _repropagate(self) -> None:
        if self._result is None:
            logger.debug('No result to repropagate!')
        else:
            self._propagate(self._result.itemIndex)

    def propagate(self, itemIndex: int) -> None:
        # FIXME initialize self._dialog.parametersView
        self._dialog.parametersView.numberOfStepsSpinBox.setRange(1, 99)  # FIXME
        self._propagate(itemIndex)

        if self._result is None:
            logger.debug('No result to show!')
        else:
            itemName = self._result.itemName
            self._dialog.setWindowTitle(f'Propagate Probe: {itemName}')
            self._dialog.open()

    def _saveResult(self) -> None:
        if self._result is None:
            logger.debug('No result to save!')
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._dialog,
            'Save Propagated Probe',
            nameFilters=self._propagator.getSaveFileFilterList(),
            selectedNameFilter=self._propagator.getSaveFileFilter())

        if filePath:
            try:
                self._propagator.savePropagatedProbe(filePath, self._result)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)
