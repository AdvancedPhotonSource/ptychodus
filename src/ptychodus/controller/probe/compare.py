from __future__ import annotations
import logging

import numpy

from PyQt5.QtCore import QAbstractItemModel

from ptychodus.api.analyze import ProbeProfileScalars
from ptychodus.api.constants import LengthUnit

from ...model.analysis import ProbeAnalyzer, ProbeComparisonReport
from ...view.probe import ProbeCompareDialog
from ...view.widgets import ExceptionDialog
from ..dashboard import (
    format_resampling,
    format_scalar,
    pixelwise_rows,
    select_pair,
    set_table_rows,
    wait_cursor,
)
from ..data import FileDialogFactory
from ..rendering import VisualizationEngine
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


def _profile_values(profile: ProbeProfileScalars) -> list[float | None]:
    """The compared figures of one probe, in the order of ``_PROFILE_LABELS``."""
    size = profile.size
    nanometer = LengthUnit.NANOMETER
    sizes: list[float | None] = (
        [None] * 5
        if size is None
        else [
            nanometer.convert(size.fwhm_major_axis_length_m),
            nanometer.convert(size.fwhm_minor_axis_length_m),
            nanometer.convert(size.rms_major_axis_length_m),
            nanometer.convert(size.rms_minor_axis_length_m),
            nanometer.convert(size.encircled_energy_diameter_m),
        ]
    )
    return [
        *sizes,
        profile.entropy.real_space_intensity_entropy,
        profile.entropy.spectral_entropy,
        float(profile.num_modes),
        profile.effective_mode_count,
        profile.purity,
        profile.von_neumann_entropy,
    ]


class ProbeCompareViewController:
    _PROFILE_LABELS = (
        'FWHM Major [nm]',
        'FWHM Minor [nm]',
        'RMS Major [nm]',
        'RMS Minor [nm]',
        'Encircled Energy [nm]',
        'Real-Space Entropy',
        'Spectral Entropy',
        'Modes',
        'Effective Modes',
        'Purity',
        'Von Neumann Entropy [nats]',
    )

    def __init__(
        self,
        analyzer: ProbeAnalyzer,
        tree_model: QAbstractItemModel,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._analyzer = analyzer
        self._dialog = ProbeCompareDialog()
        self._dialog.setWindowTitle('Compare Probes')
        self._dialog.reference_combo_box.setModel(tree_model)
        self._dialog.reference_combo_box.activated.connect(self._compare_current)
        self._dialog.test_combo_box.setModel(tree_model)
        self._dialog.test_combo_box.activated.connect(self._compare_current)

        engine = VisualizationEngine(is_complex=True)
        self._parameters_controller = VisualizationParametersController(
            engine, self._dialog.visualization_parameters_view
        )
        self._image_controllers = [
            VisualizationWidgetController(
                engine, widget, self._dialog.status_bar, file_dialog_factory
            )
            for widget in (
                self._dialog.reference_widget,
                self._dialog.test_widget,
                self._dialog.difference_widget,
            )
        ]

        reference_controller = self._image_controllers[0]

        for controller in self._image_controllers[1:]:
            reference_controller.link_to(controller)

    def compare(self, product_index: int) -> None:
        select_pair(self._dialog.reference_combo_box, self._dialog.test_combo_box, product_index)
        self._dialog.open()
        self._compare_current()

    def _compare_current(self) -> None:
        reference_index = self._dialog.reference_combo_box.currentIndex()
        test_index = self._dialog.test_combo_box.currentIndex()

        if reference_index < 0 or test_index < 0:
            self._clear()
            return

        try:
            with wait_cursor():
                report = self._analyzer.compare(reference_index, test_index)
        except Exception as err:
            logger.exception(err)
            self._clear()
            ExceptionDialog.show_exception('Compare Probes', err)
            return

        self._show(report)

    def _clear(self) -> None:
        self._dialog.registration_group.reset()
        set_table_rows(self._dialog.profile_table_view, [])
        set_table_rows(self._dialog.pixelwise_table_view, [])

        for controller in self._image_controllers:
            controller.clear_array()

        self._dialog.spectrum_plot_view.axes.clear()
        self._dialog.spectrum_plot_view.figure_canvas.draw()

    def _show(self, report: ProbeComparisonReport) -> None:
        dialog = self._dialog
        scalars = report.scalars

        dialog.fidelity_label.setText(format_scalar(scalars.normalized_fidelity, precision=6))
        dialog.rms_error_label.setText(format_scalar(scalars.normalized_rms_error, precision=4))
        dialog.probe_shift_x_label.setText(format_scalar(scalars.probe_shift_x_px))
        dialog.probe_shift_y_label.setText(format_scalar(scalars.probe_shift_y_px))
        dialog.object_shift_x_label.setText(format_scalar(scalars.object_shift_x_px))
        dialog.object_shift_y_label.setText(format_scalar(scalars.object_shift_y_px))
        dialog.shift_difference_x_label.setText(
            format_scalar(scalars.probe_shift_x_px - scalars.object_shift_x_px)
        )
        dialog.shift_difference_y_label.setText(
            format_scalar(scalars.probe_shift_y_px - scalars.object_shift_y_px)
        )
        dialog.resampled_label.setText(
            format_resampling(scalars.test_object_resampled_from, scalars.test_probe_resampled_from)
        )

        profile_rows: list[tuple[str, list[str]]] = []

        for label, reference_value, test_value in zip(
            self._PROFILE_LABELS,
            _profile_values(scalars.reference),
            _profile_values(scalars.test),
        ):
            difference = (
                None
                if reference_value is None or test_value is None
                else test_value - reference_value
            )
            profile_rows.append(
                (
                    label,
                    [
                        format_scalar(reference_value, precision=4),
                        format_scalar(test_value, precision=4),
                        format_scalar(difference, precision=3),
                    ],
                )
            )

        set_table_rows(dialog.profile_table_view, profile_rows)
        set_table_rows(
            dialog.pixelwise_table_view, pixelwise_rows(scalars.amplitude, scalars.phase)
        )

        pixel_geometry = report.comparison.reference_probe.get_pixel_geometry()
        reference_controller, test_controller, difference_controller = self._image_controllers
        reference_controller.set_array(report.reference_top_mode, pixel_geometry)
        test_controller.set_array(report.test_top_mode, pixel_geometry)
        difference_controller.set_array(
            report.test_top_mode - report.reference_top_mode, pixel_geometry
        )
        reference_controller.zoom_to_fit()

        reference_power = report.reference.relative_power
        test_power = report.test.relative_power
        num_modes = max(reference_power.size, test_power.size)
        modes = numpy.arange(num_modes)
        width = 0.4

        axes = dialog.spectrum_plot_view.axes
        axes.clear()
        axes.bar(
            modes - width / 2,
            100.0 * numpy.pad(reference_power, (0, num_modes - reference_power.size)),
            width,
            label='Reference',
        )
        axes.bar(
            modes + width / 2,
            100.0 * numpy.pad(test_power, (0, num_modes - test_power.size)),
            width,
            label='Test',
        )
        axes.set_xlabel('Orthogonal Mode')
        axes.set_ylabel('Power [%]')
        axes.set_xticks(modes)
        axes.grid(True, axis='y')
        axes.legend(loc='best', fontsize='small')
        dialog.spectrum_plot_view.figure.tight_layout()
        dialog.spectrum_plot_view.figure_canvas.draw()
