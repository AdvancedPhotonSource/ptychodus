import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import IlluminationMapper
from ...model.visualization import VisualizationEngine
from ...view.probe import IlluminationDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

logger = logging.getLogger(__name__)


class IlluminationViewController(Observer):
    def __init__(
        self,
        mapper: IlluminationMapper,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
        *,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__()
        self._mapper = mapper
        self._file_dialog_factory = file_dialog_factory
        self._dialog = IlluminationDialog()
        self._dialog.exposure_parameters_view.setVisible(is_developer_mode_enabled)
        self._dialog.exposure_quantity_view.setVisible(is_developer_mode_enabled)
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

        mapper.add_observer(self)

    def map(self, product_index: int) -> None:
        self._mapper.set_product(product_index)

        try:
            product_name = self._mapper.get_product_name()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Illumination Mapper', err)
        else:
            self._dialog.setWindowTitle(f'Illumination Map: {product_name}')
            self._dialog.open()

        try:
            self._mapper.map()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Illumination Mapper', err)

    def _save_data(self) -> None:
        title = 'Save Illumination Map'
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=self._mapper.get_save_file_filters(),
            selected_name_filter=self._mapper.get_save_file_filter(),
        )

        if file_path:
            try:
                self._mapper.save_data(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _sync_model_to_view(self) -> None:
        try:
            data = self._mapper.get_data()
        except ValueError:
            self._visualization_widget_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)
        else:
            self._visualization_widget_controller.set_array(data.photon_number, data.pixel_geometry)

    def _update(self, observable: Observable) -> None:
        if observable is self._mapper:
            self._sync_model_to_view()
