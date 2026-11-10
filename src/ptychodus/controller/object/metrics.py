from __future__ import annotations
import logging

import numpy

from PyQt5.QtCore import QAbstractItemModel

from ptychodus.api.analyze import ObjectLayerScalars
from ptychodus.api.constants import LengthUnit
from ptychodus.api.metrics import PowerSpectralDensity
from ptychodus.api.typing import ComplexArrayType

from ...model.analysis import ObjectAnalyzer, ObjectMetricsReport
from ...view.object import ObjectMetricsDialog
from ...view.widgets import ExceptionDialog
from ..dashboard import format_scalar, plot_histogram, set_table_rows, wait_cursor
from ..data import FileDialogFactory
from ..rendering import VisualizationEngine
from ..visualization import VisualizationWidgetController

logger = logging.getLogger(__name__)


class ObjectMetricsViewController:
    def __init__(
        self,
        analyzer: ObjectAnalyzer,
        tree_model: QAbstractItemModel,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._analyzer = analyzer
        self._report: ObjectMetricsReport | None = None
        self._dialog = ObjectMetricsDialog()
        self._dialog.setWindowTitle('Object Metrics')
        self._dialog.product_combo_box.setModel(tree_model)
        self._dialog.product_combo_box.activated.connect(self._analyze_current)
        self._dialog.layer_combo_box.activated.connect(self._show_layer)

        self._amplitude_controller = VisualizationWidgetController(
            VisualizationEngine(is_complex=False),
            self._dialog.amplitude_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._phase_controller = VisualizationWidgetController(
            VisualizationEngine(is_complex=False),
            self._dialog.phase_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._amplitude_controller.link_to(self._phase_controller)

    def analyze(self, product_index: int) -> None:
        self._dialog.product_combo_box.setCurrentIndex(product_index)
        self._dialog.open()
        self._analyze_current()

    def _analyze_current(self) -> None:
        product_index = self._dialog.product_combo_box.currentIndex()
        self._report = None

        if product_index >= 0:
            try:
                with wait_cursor():
                    self._report = self._analyzer.compute_metrics(product_index)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('Object Metrics', err)

        self._show_report()

    def _show_report(self) -> None:
        dialog = self._dialog
        report = self._report
        dialog.layer_combo_box.clear()

        if report is None:
            for group in (dialog.ambiguities_group, dialog.region_group):
                group.reset()

            set_table_rows(dialog.layers_table_view, [])
            self._show_layer()
            return

        scalars = report.scalars
        ambiguities = scalars.ambiguities
        dialog.phase_offset_label.setText(format_scalar(ambiguities.phase_offset_rad))
        dialog.phase_ramp_x_label.setText(
            format_scalar(1.0e-6 * ambiguities.phase_ramp_x_rad_per_m)
        )
        dialog.phase_ramp_y_label.setText(
            format_scalar(1.0e-6 * ambiguities.phase_ramp_y_rad_per_m)
        )
        dialog.region_size_label.setText(f'{scalars.region_width_px} × {scalars.region_height_px}')
        dialog.region_fraction_label.setText(format_scalar(100.0 * scalars.region_area_fraction))

        dialog.layer_combo_box.addItem('Projected')
        micrometer = LengthUnit.MICROMETER

        for number, depth_m in enumerate(scalars.layer_depths_m):
            dialog.layer_combo_box.addItem(
                f'Layer {number} ({format_scalar(micrometer.convert(depth_m), suffix=" µm")})'
            )

        has_layers = len(scalars.layers) > 1
        dialog.layers_group.setVisible(has_layers)
        set_table_rows(
            dialog.layers_table_view,
            [
                (
                    f'Layer {number}',
                    [
                        format_scalar(micrometer.convert(depth_m)),
                        format_scalar(layer.phase_deviation_rad),
                        format_scalar(layer.amplitude_mean),
                    ],
                )
                for number, (depth_m, layer) in enumerate(
                    zip(scalars.layer_depths_m, scalars.layers)
                )
            ]
            if has_layers
            else [],
        )
        self._show_layer()
        self._amplitude_controller.zoom_to_fit()

    def _selected_layer(
        self, report: ObjectMetricsReport
    ) -> tuple[ObjectLayerScalars, ComplexArrayType, PowerSpectralDensity]:
        """The scalars, standardized values and spectrum of the layer the combo box names."""
        selection = self._dialog.layer_combo_box.currentIndex()
        standardized = report.standardized_object

        if selection <= 0:
            return (
                report.scalars.projected,
                standardized.get_layers_flattened(),
                report.projected_spectrum,
            )

        number = selection - 1
        return (
            report.scalars.layers[number],
            standardized.get_layer(number),
            report.layer_spectra[number],
        )

    def _show_layer(self) -> None:
        dialog = self._dialog
        report = self._report

        for axes in (dialog.psd_axes, dialog.amplitude_histogram_axes, dialog.phase_histogram_axes):
            axes.clear()

        if report is None:
            dialog.statistics_group.reset()
            self._amplitude_controller.clear_array()
            self._phase_controller.clear_array()
            dialog.plot_view.figure_canvas.draw()
            return

        statistics, values, spectrum = self._selected_layer(report)
        dialog.amplitude_mean_label.setText(format_scalar(statistics.amplitude_mean, precision=4))
        dialog.amplitude_std_label.setText(format_scalar(statistics.amplitude_std, precision=4))
        dialog.amplitude_minimum_label.setText(
            format_scalar(statistics.amplitude_minimum, precision=4)
        )
        dialog.amplitude_maximum_label.setText(
            format_scalar(statistics.amplitude_maximum, precision=4)
        )
        dialog.phase_mean_label.setText(format_scalar(statistics.phase_mean_rad, precision=4))
        dialog.phase_deviation_label.setText(
            format_scalar(statistics.phase_deviation_rad, precision=4)
        )
        dialog.phase_peak_to_valley_label.setText(
            format_scalar(statistics.phase_peak_to_valley_rad, precision=4)
        )

        region = report.scoring_region
        pixel_geometry = report.standardized_object.get_pixel_geometry()
        amplitude = numpy.absolute(values)
        phase = numpy.angle(values)

        for controller, image in (
            (self._amplitude_controller, amplitude),
            (self._phase_controller, phase),
        ):
            controller.set_array(image, pixel_geometry)
            controller.set_region_outline(
                region.row_begin, region.row_end, region.column_begin, region.column_end
            )

        density = spectrum.power_spectral_density_m2
        dialog.psd_axes.semilogy(
            1.0e-9 * spectrum.spatial_frequency_per_m,
            numpy.where(density > 0.0, density, numpy.nan),
            '.-',
            linewidth=1.5,
        )
        dialog.psd_axes.set_xlabel('Spatial Frequency [1/nm]')
        dialog.psd_axes.set_ylabel('Power Spectral Density [m²]')
        dialog.psd_axes.grid(True, which='both')

        plot_histogram(
            dialog.amplitude_histogram_axes, region.crop(amplitude).ravel(), max_bins=100
        )
        dialog.amplitude_histogram_axes.set_xlabel('Amplitude')
        dialog.amplitude_histogram_axes.set_ylabel('Pixels')

        plot_histogram(dialog.phase_histogram_axes, region.crop(phase).ravel(), max_bins=100)
        dialog.phase_histogram_axes.set_xlabel('Phase [rad]')
        dialog.phase_histogram_axes.set_ylabel('Pixels')

        dialog.plot_view.figure.tight_layout()
        dialog.plot_view.figure_canvas.draw()
