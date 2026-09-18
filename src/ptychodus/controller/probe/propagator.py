import logging

import numpy

from PyQt5.QtWidgets import QSpinBox

from ptychodus.api.constants import LengthUnit, format_length
from ptychodus.api.probe import ProbeSizeMetrics
from ptychodus.api.propagate import PropagatedProbe, intensity
from ptychodus.api.typing import NumberArrayType, RealArrayType

from ...model.analysis import ProbePropagatorSettings, ProbePropagator
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from ..parameters import (
    LengthParameterViewController,
    SpinBoxParameterViewController,
)
from ..visualization import VisualizationWidgetController
from .focus import ProbeFocusViewController
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

        # The propagation controls share a strip with the depth slider and carry no form
        # labels of their own, so each needs a tool tip to stay self-describing.
        self._begin_coordinate_view_controller = LengthParameterViewController(
            settings.begin_coordinate_m, tool_tip='Begin Coordinate'
        )
        self._end_coordinate_view_controller = LengthParameterViewController(
            settings.end_coordinate_m, tool_tip='End Coordinate'
        )
        self._num_steps_view_controller = SpinBoxParameterViewController(
            settings.num_steps, tool_tip='Number of Steps'
        )

        # A plain spin box rather than SpinBoxParameterViewController: the range is the
        # propagated probe's mode count, and IntegerParameter bounds are immutable, so a
        # settings-backed parameter could not track it.
        self._mode_spin_box = QSpinBox()
        self._mode_spin_box.setToolTip('Incoherent Probe Mode')
        self._mode_spin_box.setRange(1, 1)
        self._mode_spin_box.setEnabled(False)

        self._dialog = ProbePropagationDialog(
            self._begin_coordinate_view_controller.get_widget(),
            self._end_coordinate_view_controller.get_widget(),
            self._num_steps_view_controller.get_widget(),
            self._mode_spin_box,
        )
        self._dialog.propagate_button.clicked.connect(self._propagate)
        self._dialog.focus_button.clicked.connect(self._analyze_focus)
        self._dialog.save_button.clicked.connect(self._save_propagated_probe)
        self._dialog.coordinate_slider.valueChanged.connect(self._update_current_coordinate)

        # The focus dialog drives the propagation plane through a callback rather than a
        # reference back to this controller, so the dependency stays one-way.
        self._focus_view_controller = ProbeFocusViewController(self._go_to_step, self._dialog)

        probe_view = self._dialog.probe_view
        probe_view.intensity_button.setChecked(True)
        probe_view.intensity_button.toggled.connect(self._handle_visualization_mode_toggled)
        self._mode_spin_box.valueChanged.connect(self._refresh_views)

        # All three views share one engine, so the XY ribbon's colorize and data-range
        # controls drive the color axis for the Z planes too.
        self._xy_image_controller = ImageController(
            engine, self._dialog.xy_view, self._dialog.status_bar, file_dialog_factory
        )
        self._zx_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.zx_view, self._dialog.status_bar, file_dialog_factory
        )
        self._zy_visualization_widget_controller = VisualizationWidgetController(
            engine, self._dialog.zy_view, self._dialog.status_bar, file_dialog_factory
        )
        self._zx_visualization_widget_controller.link_to(self._zy_visualization_widget_controller)

    def _get_num_steps(self) -> int:
        if self._propagated_probe is None:
            return self._settings.num_steps.get_value()

        return self._propagated_probe.num_steps

    def _is_complex_mode(self) -> bool:
        return self._dialog.probe_view.complex_button.isChecked()

    def _get_selected_mode(self) -> int:
        """Zero-based incoherent mode index; the spin box is one-based to match the
        `Mode N` labels in the probe tree."""
        return self._mode_spin_box.value() - 1

    def _sync_mode_spin_box(self) -> None:
        num_modes = (
            1 if self._propagated_probe is None else self._propagated_probe.num_incoherent_modes
        )

        self._mode_spin_box.blockSignals(True)
        self._mode_spin_box.setRange(1, max(1, num_modes))
        self._mode_spin_box.blockSignals(False)

        # Only the complex branch renders a single mode; the intensity branch always
        # sums over all of them, so there is nothing to pick.
        self._mode_spin_box.setEnabled(self._is_complex_mode() and num_modes > 1)

    def _handle_visualization_mode_toggled(self) -> None:
        self._sync_mode_spin_box()
        self._refresh_views()

    def _refresh_views(self) -> None:
        """Re-render all three planes from the existing propagation.

        Every incoherent mode is already in the propagated stack, so switching mode or
        visualization branch never re-propagates.
        """
        self._update_current_coordinate(self._dialog.coordinate_slider.value())
        self._update_z_planes()

    def _get_xy_array(self, step: int) -> tuple[NumberArrayType, RealArrayType]:
        """Return the XY array to display and the intensity to measure it by.

        The two differ in the complex branch: probe-size estimation needs a real
        intensity, so the metrics are always computed on ``|wavefield|²``.
        """
        probe = self._propagated_probe

        if probe is None:
            raise ValueError('No propagated probe!')

        if self._is_complex_mode():
            wavefield = probe.get_xy_wavefield(step, self._get_selected_mode())
            return wavefield, intensity(wavefield)

        xy_intensity = probe.get_xy_intensity(step)
        return xy_intensity, xy_intensity

    def _update_metrics_view(self, metrics: ProbeSizeMetrics | None) -> None:
        view = self._dialog.metrics_view

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
        view.fwhm_major_axis_label.setText(
            f'{LengthUnit.NANOMETER.convert(metrics.fwhm_major_axis_length_m):.4g}'
        )
        view.fwhm_minor_axis_label.setText(
            f'{LengthUnit.NANOMETER.convert(metrics.fwhm_minor_axis_length_m):.4g}'
        )
        view.rms_major_axis_label.setText(
            f'{LengthUnit.NANOMETER.convert(metrics.rms_major_axis_length_m):.4g}'
        )
        view.rms_minor_axis_label.setText(
            f'{LengthUnit.NANOMETER.convert(metrics.rms_minor_axis_length_m):.4g}'
        )
        view.encircled_energy_diameter_label.setText(
            f'{LengthUnit.NANOMETER.convert(metrics.encircled_energy_diameter_m):.4g}'
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
            self._xy_image_controller.clear_array()
        else:
            try:
                xy_array, xy_intensity = self._get_xy_array(step)
            except IndexError:
                self._xy_image_controller.clear_array()
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('Update Current Coordinate', err)
            else:
                pixel_geometry = self._propagator.get_pixel_geometry(self._product_index)

                if pixel_geometry is None:
                    logger.warning('Missing propagator pixel geometry!')
                else:
                    self._xy_image_controller.set_array(xy_array, pixel_geometry)
                    metrics = compute_xy_metrics(xy_intensity, pixel_geometry)

        self._update_metrics_view(metrics)
        self._update_z_indicators(step)

        self._dialog.coordinate_label.setText(format_length(lerp_value))

    def _propagate(self) -> None:
        try:
            self._propagated_probe = self._propagator.propagate(self._product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Propagate Probe', err)
        else:
            self._sync_model_to_view()

    def _go_to_step(self, step: int) -> None:
        """Move the propagation plane; the slider's own signal re-renders everything."""
        self._dialog.coordinate_slider.setValue(step)

    def _analyze_focus(self) -> None:
        probe = self._propagated_probe

        if probe is None:
            logger.warning('No propagated wavefield to analyze!')
            return

        self._focus_view_controller.analyze(probe, self._get_selected_mode())

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
        self._focus_view_controller.invalidate()
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

    def _update_z_planes(self) -> None:
        pixel_geometry = self._propagator.get_pixel_geometry(self._product_index)

        if pixel_geometry is None:
            logger.warning('Missing propagator pixel geometry!')
            return

        probe = self._propagated_probe

        if probe is None:
            self._zx_visualization_widget_controller.clear_array()
            self._zy_visualization_widget_controller.clear_array()
            self._zx_visualization_widget_controller.clear_vertical_indicator()
            self._zy_visualization_widget_controller.clear_vertical_indicator()
            return

        zx_array: NumberArrayType
        zy_array: NumberArrayType

        try:
            if self._is_complex_mode():
                mode = self._get_selected_mode()
                zx_array = probe.get_zx_wavefield(mode)
                zy_array = probe.get_zy_wavefield(mode)
            else:
                zx_array = probe.get_zx_intensity()
                zy_array = probe.get_zy_intensity()

            # vvv TODO display correct pixel geometry for the Z planes vvv
            self._zx_visualization_widget_controller.set_array(zx_array, pixel_geometry)
            self._zy_visualization_widget_controller.set_array(zy_array, pixel_geometry)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Views', err)
        else:
            self._update_z_indicators(self._dialog.coordinate_slider.value())

    def _sync_model_to_view(self) -> None:
        num_steps = self._get_num_steps()

        if num_steps > 1:
            self._dialog.coordinate_slider.setEnabled(True)
            self._dialog.coordinate_slider.setRange(0, num_steps - 1)
        else:
            self._dialog.coordinate_slider.setEnabled(False)
            self._dialog.coordinate_slider.setRange(0, 1)
            self._dialog.coordinate_slider.setValue(0)

        self._dialog.focus_button.setEnabled(self._propagated_probe is not None)
        self._sync_mode_spin_box()
        self._update_current_coordinate(self._dialog.coordinate_slider.value())
        self._update_z_planes()
