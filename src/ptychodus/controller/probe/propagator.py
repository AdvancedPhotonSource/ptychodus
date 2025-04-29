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
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._propagator = propagator
        self._file_dialog_factory = file_dialog_factory

        self._dialog = ProbePropagationDialog()
        self._dialog.propagate_button.clicked.connect(self._propagate)
        self._dialog.save_button.clicked.connect(self._save_propagated_probe)
        self._dialog.coordinate_slider.valueChanged.connect(self._update_current_coordinate)
        self._dialog.parameters_view.num_steps_spin_box.setRange(1, 999)

        self._xy_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.xy_view, self._dialog.status_bar, file_dialog_factory
        )
        self._zx_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.zx_view, self._dialog.status_bar, file_dialog_factory
        )
        self._visualization_parameters_controller = (
            VisualizationParametersController.create_instance(
                engine, self._dialog.parameters_view.visualization_parameters_view
            )
        )
        self._zy_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.zy_view, self._dialog.status_bar, file_dialog_factory
        )

        propagator.add_observer(self)
        self._sync_model_to_view()

    def _update_current_coordinate(self, step: int) -> None:
        lerp_value = 0.0

        slider = self._dialog.coordinate_slider
        upper = step - slider.minimum()
        lower = slider.maximum() - slider.minimum()

        if lower > 0:
            alpha = upper / lower
            z0 = self._propagator.get_begin_coordinate_m()
            z1 = self._propagator.get_end_coordinate_m()
            lerp_value = (1 - alpha) * z0 + alpha * z1
        else:
            logger.error('Bad slider range!')

        try:
            xy_projection = self._propagator.get_xy_projection(step)
        except (IndexError, ValueError):
            self._xy_visualization_widget_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Current Coordinate', err)
        else:
            pixel_geometry = self._propagator.get_pixel_geometry()

            if pixel_geometry is None:
                logger.warning('Missing propagator pixel geometry!')
            else:
                self._xy_visualization_widget_controller.set_array(xy_projection, pixel_geometry)

        # TODO auto-units
        lerp_value *= 1e6
        self._dialog.coordinate_label.setText(f'{lerp_value:.1f} \u00b5m')

    def _propagate(self) -> None:
        view = self._dialog.parameters_view

        try:
            self._propagator.propagate(
                num_steps=view.num_steps_spin_box.value(),
                begin_coordinate_m=float(view.begin_coordinate_widget.get_length_m()),
                end_coordinate_m=float(view.end_coordinate_widget.get_length_m()),
            )
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Propagate Probe', err)

    def launch(self, product_index: int) -> None:
        self._propagator.set_product(product_index)

        try:
            item_name = self._propagator.get_product_name()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Launch', err)
        else:
            self._dialog.setWindowTitle(f'Propagate Probe: {item_name}')
            self._dialog.open()

    def _save_propagated_probe(self) -> None:
        title = 'Save Propagated Probe'
        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=self._propagator.get_save_file_filters(),
            selected_name_filter=self._propagator.get_save_file_filter(),
        )

        if file_path:
            try:
                self._propagator.save_propagated_probe(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _sync_model_to_view(self) -> None:
        view = self._dialog.parameters_view
        view.begin_coordinate_widget.set_length_m(
            Decimal(repr(self._propagator.get_begin_coordinate_m()))
        )
        view.end_coordinate_widget.set_length_m(
            Decimal(repr(self._propagator.get_end_coordinate_m()))
        )
        view.num_steps_spin_box.setValue(self._propagator.get_num_steps())

        num_steps = self._propagator.get_num_steps()

        if num_steps > 1:
            self._dialog.coordinate_slider.setEnabled(True)
            self._dialog.coordinate_slider.setRange(0, num_steps - 1)
        else:
            self._dialog.coordinate_slider.setEnabled(False)
            self._dialog.coordinate_slider.setRange(0, 1)
            self._dialog.coordinate_slider.setValue(0)

        self._update_current_coordinate(self._dialog.coordinate_slider.value())
        pixel_geometry = self._propagator.get_pixel_geometry()

        if pixel_geometry is None:
            logger.warning('Missing propagator pixel geometry!')
            return

        try:
            # vvv TODO display correct pixel geometry for projections vvv
            self._zx_visualization_widget_controller.set_array(
                self._propagator.get_zx_projection(), pixel_geometry
            )
            self._zy_visualization_widget_controller.set_array(
                self._propagator.get_zy_projection(), pixel_geometry
            )
        except ValueError:
            self._zx_visualization_widget_controller.clear_array()
            self._zy_visualization_widget_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)

    def _update(self, observable: Observable) -> None:
        if observable is self._propagator:
            self._sync_model_to_view()
