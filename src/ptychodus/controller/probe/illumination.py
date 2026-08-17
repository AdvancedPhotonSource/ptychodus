from decimal import Decimal
import logging

from PyQt5.QtWidgets import QButtonGroup

from ptychodus.api.typing import RealArrayType

from ...model.analysis import IlluminationMap, IlluminationMapper
from ...model.visualization import VisualizationEngine
from ...view.probe import IlluminationDialog, IlluminationParametersView, IlluminationQuantityView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

logger = logging.getLogger(__name__)

_SAVE_FILE_FILTER = 'NumPy Zipped Archive (*.npz)'


class IlluminationParametersController:
    def __init__(self, view: IlluminationParametersView) -> None:
        self._view = view

        self._view.photon_flux_line_edit.setEnabled(False)
        self._view.exposure_time_line_edit.setEnabled(False)
        self._view.mass_attenuation_line_edit.setEnabled(False)

    def set_illumination_map(self, illumination_map: IlluminationMap | None) -> None:
        if illumination_map is None:
            nan = Decimal('NaN')
            self._view.photon_flux_line_edit.set_value(nan)
            self._view.exposure_time_line_edit.set_value(nan)
            self._view.mass_attenuation_line_edit.set_value(nan)
            return

        self._view.photon_flux_line_edit.set_value(Decimal(repr(illumination_map.photon_flux_Hz)))
        self._view.exposure_time_line_edit.set_value(
            Decimal(repr(illumination_map.exposure_time_s))
        )
        self._view.mass_attenuation_line_edit.set_value(
            Decimal(repr(illumination_map.mass_attenuation_m2_kg))
        )


class IlluminationQuantityController:
    def __init__(
        self,
        view: IlluminationQuantityView,
        widget_controller: VisualizationWidgetController,
    ) -> None:
        self._view = view
        self._widget_controller = widget_controller
        self._illumination_map: IlluminationMap | None = None

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

        self._button_group.buttonClicked.connect(self._render_current_quantity)

    def set_illumination_map(self, illumination_map: IlluminationMap | None) -> None:
        self._illumination_map = illumination_map
        self._render_current_quantity()

    def _render_current_quantity(self) -> None:
        illumination_map = self._illumination_map

        if illumination_map is None:
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
        self._illumination_map: IlluminationMap | None = None
        self._dialog = IlluminationDialog()

        self._parameters_controller = IlluminationParametersController(self._dialog.parameters_view)
        self._visualization_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.visualization_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._quantity_controller = IlluminationQuantityController(
            self._dialog.quantity_view, self._visualization_widget_controller
        )
        self._visualization_parameters_controller = VisualizationParametersController(
            engine, self._dialog.visualization_parameters_view
        )
        self._dialog.save_button.clicked.connect(self._save_data)

    def map(self, product_index: int) -> None:
        self._illumination_map = None
        self._sync_subcontrollers()

        try:
            product_name = self._mapper.get_product_name(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Illumination Mapper', err)
            return

        self._dialog.setWindowTitle(f'Illumination Map: {product_name}')
        self._dialog.open()

        try:
            self._illumination_map = self._mapper.map(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Illumination Mapper', err)

        self._sync_subcontrollers()

    def _sync_subcontrollers(self) -> None:
        self._parameters_controller.set_illumination_map(self._illumination_map)
        self._quantity_controller.set_illumination_map(self._illumination_map)

    def _save_data(self) -> None:
        if self._illumination_map is None:
            logger.warning('No illumination map to save!')
            return

        title = 'Save Illumination Map'
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=[_SAVE_FILE_FILTER],
            selected_name_filter=_SAVE_FILE_FILTER,
        )

        if file_path:
            try:
                self._illumination_map.save_npz(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)
