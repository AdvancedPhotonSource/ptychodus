from decimal import Decimal
import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import ProbePropagator
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

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
        self._dialog.propagate_button.clicked.connect(self._propagate)
        self._dialog.save_button.clicked.connect(self._savePropagatedProbe)
        self._dialog.coordinate_slider.valueChanged.connect(self._updateCurrentCoordinate)
        self._dialog.parameters_view.num_steps_spin_box.setRange(1, 999)

        self._xyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.xy_view, self._dialog.status_bar, fileDialogFactory
        )
        self._zxVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zx_view, self._dialog.status_bar, fileDialogFactory
        )
        self._visualizationParametersController = VisualizationParametersController.create_instance(
            engine, self._dialog.parameters_view.visualization_parameters_view
        )
        self._zyVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.zy_view, self._dialog.status_bar, fileDialogFactory
        )

        propagator.add_observer(self)
        self._syncModelToView()

    def _updateCurrentCoordinate(self, step: int) -> None:
        lerpValue = 0.0

        slider = self._dialog.coordinate_slider
        upper = step - slider.minimum()
        lower = slider.maximum() - slider.minimum()

        if lower > 0:
            alpha = upper / lower
            z0 = self._propagator.getBeginCoordinateInMeters()
            z1 = self._propagator.getEndCoordinateInMeters()
            lerpValue = (1 - alpha) * z0 + alpha * z1
        else:
            logger.error('Bad slider range!')

        try:
            xyProjection = self._propagator.getXYProjection(step)
        except (IndexError, ValueError):
            self._xyVisualizationWidgetController.clearArray()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Current Coordinate', err)
        else:
            pixelGeometry = self._propagator.getPixelGeometry()

            if pixelGeometry is None:
                logger.warning('Missing propagator pixel geometry!')
            else:
                self._xyVisualizationWidgetController.set_array(xyProjection, pixelGeometry)

        # TODO auto-units
        lerpValue *= 1e6
        self._dialog.coordinate_label.setText(f'{lerpValue:.1f} \u00b5m')

    def _propagate(self) -> None:
        view = self._dialog.parameters_view

        try:
            self._propagator.propagate(
                numberOfSteps=view.num_steps_spin_box.value(),
                beginCoordinateInMeters=float(view.begin_coordinate_widget.getLengthInMeters()),
                endCoordinateInMeters=float(view.end_coordinate_widget.getLengthInMeters()),
            )
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Propagate Probe', err)

    def launch(self, productIndex: int) -> None:
        self._propagator.setProduct(productIndex)

        try:
            itemName = self._propagator.getProductName()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Launch', err)
        else:
            self._dialog.setWindowTitle(f'Propagate Probe: {itemName}')
            self._dialog.open()

    def _savePropagatedProbe(self) -> None:
        title = 'Save Propagated Probe'
        filePath, nameFilter = self._fileDialogFactory.get_save_file_path(
            self._dialog,
            title,
            name_filters=self._propagator.getSaveFileFilterList(),
            selected_name_filter=self._propagator.getSaveFileFilter(),
        )

        if filePath:
            try:
                self._propagator.savePropagatedProbe(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _syncModelToView(self) -> None:
        view = self._dialog.parameters_view
        view.begin_coordinate_widget.setLengthInMeters(
            Decimal.from_float(self._propagator.getBeginCoordinateInMeters())
        )
        view.end_coordinate_widget.setLengthInMeters(
            Decimal.from_float(self._propagator.getEndCoordinateInMeters())
        )
        view.num_steps_spin_box.setValue(self._propagator.getNumberOfSteps())

        numberOfSteps = self._propagator.getNumberOfSteps()

        if numberOfSteps > 1:
            self._dialog.coordinate_slider.setEnabled(True)
            self._dialog.coordinate_slider.setRange(0, numberOfSteps - 1)
        else:
            self._dialog.coordinate_slider.setEnabled(False)
            self._dialog.coordinate_slider.setRange(0, 1)
            self._dialog.coordinate_slider.setValue(0)

        self._updateCurrentCoordinate(self._dialog.coordinate_slider.value())
        pixelGeometry = self._propagator.getPixelGeometry()

        if pixelGeometry is None:
            logger.warning('Missing propagator pixel geometry!')
            return

        try:
            self._zxVisualizationWidgetController.set_array(
                self._propagator.getZXProjection(), pixelGeometry
            )
            self._zyVisualizationWidgetController.set_array(
                self._propagator.getZYProjection(), pixelGeometry
            )
        except ValueError:
            self._zxVisualizationWidgetController.clearArray()
            self._zyVisualizationWidgetController.clearArray()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)

    def _update(self, observable: Observable) -> None:
        if observable is self._propagator:
            self._syncModelToView()
