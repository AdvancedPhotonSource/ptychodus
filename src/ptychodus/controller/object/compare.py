from __future__ import annotations
import logging

import numpy

from PyQt5.QtCore import QAbstractItemModel

from ptychodus.api.constants import LengthUnit
from ptychodus.api.metrics import PowerSpectralDensity, SpectralSNRConvention
from ptychodus.api.typing import RealArrayType

from ...model.analysis import ObjectAnalyzer, ObjectComparisonReport
from ...view.object import ObjectCompareDialog
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


def _positive_or_nan(values: RealArrayType) -> RealArrayType:
    """Hide non-positive values, which a log axis cannot draw, as gaps."""
    return numpy.where(values > 0.0, values, numpy.nan)


class ObjectCompareViewController:
    def __init__(
        self,
        analyzer: ObjectAnalyzer,
        tree_model: QAbstractItemModel,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._analyzer = analyzer
        self._dialog = ObjectCompareDialog()
        self._dialog.setWindowTitle('Compare Objects')
        self._dialog.reference_combo_box.setModel(tree_model)
        self._dialog.reference_combo_box.activated.connect(self._compare_current)
        self._dialog.test_combo_box.setModel(tree_model)
        self._dialog.test_combo_box.activated.connect(self._compare_current)

        for convention, label in (
            (SpectralSNRConvention.HALF_SPLIT, 'Half-split of one scan'),
            (SpectralSNRConvention.REPEAT, 'Repeat scans'),
        ):
            self._dialog.convention_combo_box.addItem(label, convention)

        self._dialog.convention_combo_box.activated.connect(self._compare_current)

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

        convention = self._dialog.convention_combo_box.currentData()

        try:
            with wait_cursor():
                report = self._analyzer.compare(reference_index, test_index, convention=convention)
        except Exception as err:
            logger.exception(err)
            self._clear()
            ExceptionDialog.show_exception('Compare Objects', err)
            return

        self._show(report)

    def _clear(self) -> None:
        dialog = self._dialog
        dialog.resolution_group.reset()
        dialog.registration_group.reset()
        set_table_rows(dialog.pixelwise_table_view, [])

        for controller in self._image_controllers:
            controller.clear_array()

        for axes in (dialog.frc_axes, dialog.ssnr_axes, dialog.psd_axes):
            axes.clear()

        dialog.plot_view.figure_canvas.draw()

    def _show(self, report: ObjectComparisonReport) -> None:
        dialog = self._dialog
        scalars = report.scalars
        nanometer = LengthUnit.NANOMETER

        dialog.resolution_at_snr_label.setText(
            format_scalar(nanometer.convert(scalars.resolution_m_at_snr))
        )
        dialog.resolution_at_half_bit_label.setText(
            format_scalar(nanometer.convert(scalars.resolution_m_at_half_bit))
        )
        dialog.area_under_curve_label.setText(format_scalar(scalars.area_under_curve))
        dialog.average_snr_label.setText(format_scalar(scalars.average_signal_to_noise_ratio))

        ambiguities = scalars.ambiguities
        dialog.shift_x_label.setText(format_scalar(scalars.shift_x_px))
        dialog.shift_y_label.setText(format_scalar(scalars.shift_y_px))
        dialog.scale_label.setText(format_scalar(ambiguities.object_scale_factor, precision=4))
        dialog.phase_offset_label.setText(format_scalar(ambiguities.phase_offset_rad))
        dialog.phase_ramp_x_label.setText(
            format_scalar(1.0e-6 * ambiguities.phase_ramp_x_rad_per_m)
        )
        dialog.phase_ramp_y_label.setText(
            format_scalar(1.0e-6 * ambiguities.phase_ramp_y_rad_per_m)
        )
        dialog.resampled_label.setText(
            format_resampling(scalars.test_object_resampled_from, scalars.test_probe_resampled_from)
        )
        set_table_rows(
            dialog.pixelwise_table_view, pixelwise_rows(scalars.amplitude, scalars.phase)
        )

        region = report.scoring_region
        reference = region.crop(report.comparison.reference_complex)
        test = region.crop(report.comparison.test_complex)
        pixel_geometry = report.comparison.pixel_geometry
        reference_controller, test_controller, difference_controller = self._image_controllers
        reference_controller.set_array(reference, pixel_geometry)
        test_controller.set_array(test, pixel_geometry)
        difference_controller.set_array(test - reference, pixel_geometry)
        reference_controller.zoom_to_fit()

        frc = report.fourier_ring_correlation
        frequency_per_nm = 1.0e-9 * frc.spatial_frequency_per_m

        dialog.frc_axes.clear()
        dialog.frc_axes.plot(frequency_per_nm, frc.correlation, '.-', linewidth=1.5, label='FRC')
        dialog.frc_axes.plot(
            frequency_per_nm,
            frc.get_bit_threshold_curve(0.5),
            '--',
            linewidth=1.0,
            label='½-bit threshold',
        )
        dialog.frc_axes.set_xlabel('Spatial Frequency [1/nm]')
        dialog.frc_axes.set_ylabel('Fourier Ring Correlation')
        dialog.frc_axes.grid(True)
        dialog.frc_axes.legend(loc='best', fontsize='small')

        dialog.ssnr_axes.clear()
        dialog.ssnr_axes.semilogy(
            frequency_per_nm,
            _positive_or_nan(frc.get_spectral_signal_to_noise_ratio()),
            '.-',
            linewidth=1.5,
        )
        dialog.ssnr_axes.axhline(scalars.snr_threshold, linestyle='--', linewidth=1.0)
        dialog.ssnr_axes.set_xlabel('Spatial Frequency [1/nm]')
        dialog.ssnr_axes.set_ylabel('Spectral SNR')
        dialog.ssnr_axes.grid(True, which='both')

        dialog.psd_axes.clear()
        spectra: tuple[tuple[str, PowerSpectralDensity], ...] = (
            ('Reference', report.reference_spectrum),
            ('Test', report.test_spectrum),
        )

        for label, spectrum in spectra:
            dialog.psd_axes.semilogy(
                1.0e-9 * spectrum.spatial_frequency_per_m,
                _positive_or_nan(spectrum.power_spectral_density_m2),
                '.-',
                linewidth=1.5,
                label=label,
            )

        dialog.psd_axes.set_xlabel('Spatial Frequency [1/nm]')
        dialog.psd_axes.set_ylabel('Power Spectral Density [m²]')
        dialog.psd_axes.grid(True, which='both')
        dialog.psd_axes.legend(loc='best', fontsize='small')

        dialog.plot_view.figure.tight_layout()
        dialog.plot_view.figure_canvas.draw()
