import logging

import numpy

from ptychodus.api.probe import ProbeSizeMetrics
from ptychodus.api.propagator import PropagatedProbe

from ...model.analysis import ProbePropagatorSettings, ProbePropagator
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..parametric import (
    LengthWidgetParameterViewController,
    SpinBoxParameterViewController,
)
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)
from .metrics import compute_xy_metrics

logger = logging.getLogger(__name__)

_SAVE_FILE_FILTER = 'NumPy Zipped Archive (*.npz)'


class ProbePropagationViewController:
    def __init__(
        self,
        propagator: ProbePropagator,
        settings: ProbePropagatorSettings,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._propagator = propagator
        self._settings = settings
        self._file_dialog_factory = file_dialog_factory

        self._product_index = -1
        self._propagated_probe: PropagatedProbe | None = None

        self._begin_coordinate_view_controller = LengthWidgetParameterViewController(
            settings.begin_coordinate_m, is_signed=True
        )
        self._end_coordinate_view_controller = LengthWidgetParameterViewController(
            settings.end_coordinate_m, is_signed=True
        )
        self._num_steps_view_controller = SpinBoxParameterViewController(settings.num_steps)

        self._dialog = ProbePropagationDialog(
            self._begin_coordinate_view_controller.get_widget(),
            self._end_coordinate_view_controller.get_widget(),
            self._num_steps_view_controller.get_widget(),
        )
        self._dialog.propagate_button.clicked.connect(self._propagate)
        self._dialog.save_button.clicked.connect(self._save_propagated_probe)
        self._dialog.coordinate_slider.valueChanged.connect(self._update_current_coordinate)

        self._xy_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.xy_view, self._dialog.status_bar, file_dialog_factory
        )
        self._zx_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.zx_view, self._dialog.status_bar, file_dialog_factory
        )
        self._visualization_parameters_controller = VisualizationParametersController(
            engine, self._dialog.parameters_view.visualization_parameters_view
        )
        self._zy_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.zy_view, self._dialog.status_bar, file_dialog_factory
        )

    def _get_num_steps(self) -> int:
        if self._propagated_probe is None:
            return self._settings.num_steps.get_value()

        return self._propagated_probe.num_steps

    def _update_metrics_view(self, metrics: ProbeSizeMetrics | None) -> None:
        view = self._dialog.parameters_view.metrics_view

        if metrics is None:
            for label in (
                view.major_axis_tilt_label,
                view.minor_axis_tilt_label,
                view.fwhm_major_axis_label,
                view.fwhm_minor_axis_label,
                view.rms_major_axis_label,
                view.rms_minor_axis_label,
                view.encircled_energy_diameter_label,
            ):
                label.setText('N/A')
            return

        view.major_axis_tilt_label.setText(f'{numpy.rad2deg(metrics.major_axis_tilt_rad):.4g}')
        view.minor_axis_tilt_label.setText(f'{numpy.rad2deg(metrics.minor_axis_tilt_rad):.4g}')
        view.fwhm_major_axis_label.setText(f'{metrics.fwhm_major_axis_length_m * 1e9:.4g}')
        view.fwhm_minor_axis_label.setText(f'{metrics.fwhm_minor_axis_length_m * 1e9:.4g}')
        view.rms_major_axis_label.setText(f'{metrics.rms_major_axis_length_m * 1e9:.4g}')
        view.rms_minor_axis_label.setText(f'{metrics.rms_minor_axis_length_m * 1e9:.4g}')
        view.encircled_energy_diameter_label.setText(
            f'{metrics.encircled_energy_diameter_m * 1e9:.4g}'
        )

    def _update_z_indicators(self, step: int) -> None:
        if self._get_num_steps() > 0:
            indicator_x = float(step) + 0.5
            self._zx_visualization_widget_controller.set_vertical_indicator(indicator_x)
            self._zy_visualization_widget_controller.set_vertical_indicator(indicator_x)
        else:
            self._zx_visualization_widget_controller.clear_vertical_indicator()
            self._zy_visualization_widget_controller.clear_vertical_indicator()

    def _update_current_coordinate(self, step: int) -> None:
        lerp_value = 0.0

        slider = self._dialog.coordinate_slider
        upper = step - slider.minimum()
        lower = slider.maximum() - slider.minimum()

        if lower > 0:
            alpha = upper / lower
            settings = self._settings
            z0 = settings.begin_coordinate_m.get_value()
            z1 = settings.end_coordinate_m.get_value()
            lerp_value = (1 - alpha) * z0 + alpha * z1
        else:
            logger.error('Bad slider range!')

        metrics: ProbeSizeMetrics | None = None

        if self._propagated_probe is None:
            self._xy_visualization_widget_controller.clear_array()
        else:
            try:
                xy_projection = self._propagated_probe.get_xy_projection(step)
            except IndexError:
                self._xy_visualization_widget_controller.clear_array()
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('Update Current Coordinate', err)
            else:
                pixel_geometry = self._propagator.get_pixel_geometry(self._product_index)

                if pixel_geometry is None:
                    logger.warning('Missing propagator pixel geometry!')
                else:
                    self._xy_visualization_widget_controller.set_array(
                        xy_projection, pixel_geometry
                    )
                    metrics = compute_xy_metrics(xy_projection, pixel_geometry)

        self._update_metrics_view(metrics)
        self._update_z_indicators(step)

        # TODO auto-units
        lerp_value *= 1e6
        self._dialog.coordinate_label.setText(f'{lerp_value:.1f} µm')

    def _propagate(self) -> None:
        try:
            self._propagated_probe = self._propagator.propagate(self._product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Propagate Probe', err)
        else:
            self._sync_model_to_view()

    def launch(self, product_index: int) -> None:
        self._product_index = product_index
        self._propagated_probe = None

        try:
            item_name = self._propagator.get_product_name(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Launch', err)
            return

        self._dialog.setWindowTitle(f'Propagate Probe: {item_name}')
        self._sync_model_to_view()
        self._dialog.open()

    def _save_propagated_probe(self) -> None:
        if self._propagated_probe is None:
            logger.warning('No propagated wavefield to save!')
            return

        title = 'Save Propagated Probe'
        file_path, _name_filter = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=[_SAVE_FILE_FILTER],
            selected_name_filter=_SAVE_FILE_FILTER,
        )

        if file_path:
            try:
                self._propagated_probe.save_npz(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _sync_model_to_view(self) -> None:
        num_steps = self._get_num_steps()

        if num_steps > 1:
            self._dialog.coordinate_slider.setEnabled(True)
            self._dialog.coordinate_slider.setRange(0, num_steps - 1)
        else:
            self._dialog.coordinate_slider.setEnabled(False)
            self._dialog.coordinate_slider.setRange(0, 1)
            self._dialog.coordinate_slider.setValue(0)

        self._update_current_coordinate(self._dialog.coordinate_slider.value())
        pixel_geometry = self._propagator.get_pixel_geometry(self._product_index)

        if pixel_geometry is None:
            logger.warning('Missing propagator pixel geometry!')
            return

        if self._propagated_probe is None:
            self._zx_visualization_widget_controller.clear_array()
            self._zy_visualization_widget_controller.clear_array()
            self._zx_visualization_widget_controller.clear_vertical_indicator()
            self._zy_visualization_widget_controller.clear_vertical_indicator()
            return

        try:
            # vvv TODO display correct pixel geometry for projections vvv
            self._zx_visualization_widget_controller.set_array(
                self._propagated_probe.get_zx_projection(), pixel_geometry
            )
            self._zy_visualization_widget_controller.set_array(
                self._propagated_probe.get_zy_projection(), pixel_geometry
            )
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)
        else:
            self._update_z_indicators(self._dialog.coordinate_slider.value())
