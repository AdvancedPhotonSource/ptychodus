from decimal import Decimal
import logging

from PyQt5.QtWidgets import QButtonGroup

from ptychodus.api.common import RealArrayType
from ptychodus.api.observer import Observable, Observer

from ...model.analysis import IlluminationMapper
from ...model.visualization import VisualizationEngine
from ...view.probe import IlluminationDialog, IlluminationParametersView, IlluminationQuantityView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

logger = logging.getLogger(__name__)


class IlluminationParametersController(Observer):
    def __init__(self, mapper: IlluminationMapper, view: IlluminationParametersView) -> None:
        super().__init__()
        self._mapper = mapper
        self._view = view

        self._view.photon_flux_line_edit.setEnabled(False)
        self._view.exposure_time_line_edit.setEnabled(False)
        self._view.mass_attenuation_line_edit.setEnabled(False)

        self._mapper.add_observer(self)

    def _update_parameters(self) -> None:
        try:
            illumination_map = self._mapper.get_illumination_map()
        except ValueError:
            pass
        else:
            self._view.photon_flux_line_edit.set_value(
                Decimal(repr(illumination_map.photon_flux_Hz))
            )
            self._view.exposure_time_line_edit.set_value(
                Decimal(repr(illumination_map.exposure_time_s))
            )
            self._view.mass_attenuation_line_edit.set_value(
                Decimal(repr(illumination_map.mass_attenuation_m2_kg))
            )
            return

        nan = Decimal('NaN')
        self._view.exposure_time_line_edit.set_value(nan)
        self._view.mass_attenuation_line_edit.set_value(nan)

    def _update(self, observable: Observable) -> None:
        if observable is self._mapper:
            self._update_parameters()


class IlluminationQuantityController(Observer):
    def __init__(
        self,
        mapper: IlluminationMapper,
        view: IlluminationQuantityView,
        widget_controller: VisualizationWidgetController,
    ) -> None:
        super().__init__()
        self._mapper = mapper
        self._view = view
        self._widget_controller = widget_controller

        self._button_group = QButtonGroup()
        self._button_group.addButton(view.photon_number_button)
        self._button_group.addButton(view.photon_fluence_button)
        self._button_group.addButton(view.photon_fluence_rate_button)
        self._button_group.addButton(view.energy_fluence_button)
        self._button_group.addButton(view.energy_fluence_rate_button)
        self._button_group.addButton(view.dose_button)
        self._button_group.addButton(view.dose_rate_button)
        self._button_group.setExclusive(True)
        view.photon_number_button.setChecked(True)

        self._button_group.buttonClicked.connect(self._update_quantity)
        self._mapper.add_observer(self)

    def _update_quantity(self) -> None:
        try:
            illumination_map = self._mapper.get_illumination_map()
        except ValueError:
            self._widget_controller.clear_array()
            return

        quantity: RealArrayType | None = None

        match self._button_group.checkedButton():
            case self._view.photon_number_button:
                quantity = illumination_map.photon_number
            case self._view.photon_fluence_button:
                quantity = illumination_map.photon_fluence_1_m2
            case self._view.photon_fluence_rate_button:
                quantity = illumination_map.photon_fluence_rate_Hz_m2
            case self._view.energy_fluence_button:
                quantity = illumination_map.energy_fluence_J_m2
            case self._view.energy_fluence_rate_button:
                quantity = illumination_map.energy_fluence_rate_W_m2
            case self._view.dose_button:
                quantity = illumination_map.dose_Gy
            case self._view.dose_rate_button:
                quantity = illumination_map.dose_rate_Gy_s

        if quantity is None:
            self._widget_controller.clear_array()
        else:
            self._widget_controller.set_array(quantity, illumination_map.pixel_geometry)

    def _update(self, observable: Observable) -> None:
        if observable is self._mapper:
            self._update_quantity()


class IlluminationViewController:
    def __init__(
        self,
        mapper: IlluminationMapper,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._mapper = mapper
        self._file_dialog_factory = file_dialog_factory
        self._dialog = IlluminationDialog()

        self._parameters_controller = IlluminationParametersController(
            mapper, self._dialog.parameters_view
        )
        self._visualization_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.visualization_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._quantity_controller = IlluminationQuantityController(
            mapper, self._dialog.quantity_view, self._visualization_widget_controller
        )
        self._visualization_parameters_controller = VisualizationParametersController(
            engine, self._dialog.visualization_parameters_view
        )
        self._dialog.save_button.clicked.connect(self._save_data)

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
