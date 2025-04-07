import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import STXMSimulator
from ...model.visualization import VisualizationEngine
from ...view.probe import STXMDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

logger = logging.getLogger(__name__)


class STXMViewController(Observer):
    def __init__(
        self,
        simulator: STXMSimulator,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._simulator = simulator
        self._file_dialog_factory = file_dialog_factory
        self._dialog = STXMDialog()
        self._dialog.save_button.clicked.connect(self._save_data)
        self._visualization_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.visualization_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._visualization_parameters_controller = (
            VisualizationParametersController.create_instance(
                engine, self._dialog.visualization_parameters_view
            )
        )

        simulator.add_observer(self)

    def simulate(self, product_index: int) -> None:
        self._simulator.set_product(product_index)

        try:
            product_name = self._simulator.get_product_name()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Simulate STXM', err)
        else:
            self._dialog.setWindowTitle(f'Simulate STXM: {product_name}')
            self._dialog.open()

        try:
            self._simulator.simulate()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Simulate STXM', err)

    def _save_data(self) -> None:
        title = 'Save STXM Data'
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=self._simulator.get_save_file_filters(),
            selected_name_filter=self._simulator.get_save_file_filter(),
        )

        if file_path:
            try:
                self._simulator.save_data(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _sync_model_to_view(self) -> None:
        try:
            data = self._simulator.get_data()
        except ValueError:
            self._visualization_widget_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)
        else:
            self._visualization_widget_controller.set_array(data.intensity, data.pixel_geometry)

    def _update(self, observable: Observable) -> None:
        if observable is self._simulator:
            self._sync_model_to_view()
