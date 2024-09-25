from decimal import Decimal
import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import ProbePropagator
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class ProbePropagationViewController(Observer):
    def __init__(
        self,
        propagator: ProbePropagator,
        engine: VisualizationEngine,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._propagator = propagator
        self._fileDialogFactory = fileDialogFactory

        self._dialog = ProbePropagationDialog()
        self._dialog.propagateButton.clicked.connect(self._propagate)
        self._dialog.saveButton.clicked.connect(self._savePropagatedProbe)
        self._dialog.coordinateSlider.valueChanged.connect(self._updateCurrentCoordinate)
        self._dialog.parametersView.numberOfStepsSpinBox.setRange(1, 999)

        self._xyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.xyView, self._dialog.statusBar, fileDialogFactory
        )
        self._zxVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zxView, self._dialog.statusBar, fileDialogFactory
        )
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.parametersView.visualizationParametersView
        )
        self._zyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zyView, self._dialog.statusBar, fileDialogFactory
        )

        propagator.addObserver(self)
        self._syncModelToView()

    def _updateCurrentCoordinate(self, step: int) -> None:
        lerpValue = Decimal()

        slider = self._dialog.coordinateSlider
        upper = Decimal(step - slider.minimum())
        lower = Decimal(slider.maximum() - slider.minimum())

        if lower > 0:
            alpha = upper / lower
            z0 = self._propagator.getBeginCoordinateInMeters()
            z1 = self._propagator.getEndCoordinateInMeters()
            lerpValue = (1 - alpha) * z0 + alpha * z1
        else:
            logger.error("Bad slider range!")

        try:
            xyProjection = self._propagator.getXYProjection(step)
        except (IndexError, ValueError):
            self._xyVisualizationWidgetController.clearArray()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException("Update Current Coordinate", err)
        else:
            self._xyVisualizationWidgetController.setArray(
                xyProjection, self._propagator.getPixelGeometry()
            )

        # TODO auto-units
        lerpValue *= Decimal("1e6")
        self._dialog.coordinateLabel.setText(f"{lerpValue:.1f} \u00b5m")

    def _propagate(self) -> None:
        view = self._dialog.parametersView

        try:
            self._propagator.propagate(
                numberOfSteps=view.numberOfStepsSpinBox.value(),
                beginCoordinateInMeters=view.beginCoordinateWidget.getLengthInMeters(),
                endCoordinateInMeters=view.endCoordinateWidget.getLengthInMeters(),
            )
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException("Propagate Probe", err)

    def launch(self, productIndex: int) -> None:
        self._propagator.setProduct(productIndex)

        try:
            itemName = self._propagator.getProductName()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException("Launch", err)
        else:
            self._dialog.setWindowTitle(f"Propagate Probe: {itemName}")
            self._dialog.open()

    def _savePropagatedProbe(self) -> None:
        title = "Save Propagated Probe"
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._dialog,
            title,
            nameFilters=self._propagator.getSaveFileFilterList(),
            selectedNameFilter=self._propagator.getSaveFileFilter(),
        )

        if filePath:
            try:
                self._propagator.savePropagatedProbe(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException(title, err)

    def _syncModelToView(self) -> None:
        view = self._dialog.parametersView
        view.beginCoordinateWidget.setLengthInMeters(self._propagator.getBeginCoordinateInMeters())
        view.endCoordinateWidget.setLengthInMeters(self._propagator.getEndCoordinateInMeters())
        view.numberOfStepsSpinBox.setValue(self._propagator.getNumberOfSteps())

        numberOfSteps = self._propagator.getNumberOfSteps()

        if numberOfSteps > 1:
            self._dialog.coordinateSlider.setEnabled(True)
            self._dialog.coordinateSlider.setRange(0, numberOfSteps - 1)
        else:
            self._dialog.coordinateSlider.setEnabled(False)
            self._dialog.coordinateSlider.setRange(0, 1)
            self._dialog.coordinateSlider.setValue(0)

        self._updateCurrentCoordinate(self._dialog.coordinateSlider.value())

        try:
            self._zxVisualizationWidgetController.setArray(
                self._propagator.getZXProjection(), self._propagator.getPixelGeometry()
            )
            self._zyVisualizationWidgetController.setArray(
                self._propagator.getZYProjection(), self._propagator.getPixelGeometry()
            )
        except ValueError:
            self._zxVisualizationWidgetController.clearArray()
            self._zyVisualizationWidgetController.clearArray()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException("Update Views", err)

    def update(self, observable: Observable) -> None:
        if observable is self._propagator:
            self._syncModelToView()
