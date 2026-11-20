from __future__ import annotations
import logging

from PyQt5.QtCore import QAbstractItemModel

from ptychodus.api.constants import LengthUnit

from ...model.analysis import ProbePositionsAnalyzer, ProbePositionsMetricsReport
from ...view.probe_positions import ProbePositionsMetricsDialog
from ...view.widgets import ExceptionDialog
from ..dashboard import format_scalar, plot_histogram, wait_cursor

logger = logging.getLogger(__name__)


def _format_index_list(indexes: list[str], *, limit: int = 8) -> str:
    """Join the first *limit* entries of *indexes*, noting how many were left out."""
    shown = ', '.join(indexes[:limit])
    return shown if len(indexes) <= limit else f'{shown}, … ({len(indexes) - limit} more)'


class ProbePositionsMetricsViewController:
    def __init__(self, analyzer: ProbePositionsAnalyzer, table_model: QAbstractItemModel) -> None:
        self._analyzer = analyzer
        self._dialog = ProbePositionsMetricsDialog()
        self._dialog.setWindowTitle('Probe Position Metrics')
        self._dialog.product_combo_box.setModel(table_model)
        self._dialog.product_combo_box.activated.connect(self._analyze_current)

    def analyze(self, product_index: int) -> None:
        self._dialog.product_combo_box.setCurrentIndex(product_index)
        self._dialog.open()
        self._analyze_current()

    def _analyze_current(self) -> None:
        product_index = self._dialog.product_combo_box.currentIndex()

        if product_index < 0:
            self._clear()
            return

        try:
            with wait_cursor():
                report = self._analyzer.compute_metrics(product_index)
        except Exception as err:
            logger.exception(err)
            self._clear()
            ExceptionDialog.show_exception('Probe Position Metrics', err)
            return

        self._show(report)

    def _clear(self) -> None:
        dialog = self._dialog

        for group in (
            dialog.indexes_group,
            dialog.extent_group,
            dialog.step_group,
            dialog.i0_group,
        ):
            group.reset()

        dialog.probe_positions_plot_view.figure.clear()
        dialog.probe_positions_plot_view.figure_canvas.draw()

        for axes in (dialog.step_histogram_axes, dialog.i0_axes):
            axes.clear()

        dialog.plot_view.figure_canvas.draw()

    def _show(self, report: ProbePositionsMetricsReport) -> None:
        dialog = self._dialog
        scalars = report.scalars
        micrometer = LengthUnit.MICROMETER
        nanometer = LengthUnit.NANOMETER

        dialog.num_positions_label.setText(str(scalars.num_positions))
        dialog.num_distinct_label.setText(str(scalars.num_distinct_indexes))
        dialog.index_range_label.setText(f'{scalars.index_minimum} – {scalars.index_maximum}')
        dialog.duplicated_label.setText(str(scalars.num_duplicated_indexes))
        dialog.duplicated_label.setToolTip(
            _format_index_list([str(index) for index in report.duplicated_indexes])
        )
        dialog.missing_label.setText(
            f'{scalars.num_missing_indexes} in {scalars.num_index_gaps} gap(s)'
        )
        dialog.missing_label.setToolTip(
            _format_index_list(
                [
                    str(first) if first == last else f'{first}–{last}'
                    for first, last in report.index_gaps
                ]
            )
        )

        dialog.width_label.setText(format_scalar(micrometer.convert(scalars.width_m), precision=4))
        dialog.height_label.setText(
            format_scalar(micrometer.convert(scalars.height_m), precision=4)
        )
        dialog.center_x_label.setText(
            format_scalar(micrometer.convert(scalars.center_x_m), precision=4)
        )
        dialog.center_y_label.setText(
            format_scalar(micrometer.convert(scalars.center_y_m), precision=4)
        )
        square_micrometers_per_square_meter = 1.0e12
        dialog.hull_area_label.setText(
            format_scalar(square_micrometers_per_square_meter * scalars.convex_hull_area_m2)
        )
        dialog.density_label.setText(
            format_scalar(scalars.density_per_m2 / square_micrometers_per_square_meter)
        )

        dialog.step_mean_label.setText(
            format_scalar(nanometer.convert(scalars.nearest_neighbor_mean_m), precision=4)
        )
        dialog.step_median_label.setText(
            format_scalar(nanometer.convert(scalars.nearest_neighbor_median_m), precision=4)
        )
        dialog.step_minimum_label.setText(
            format_scalar(nanometer.convert(scalars.nearest_neighbor_minimum_m), precision=4)
        )
        dialog.step_maximum_label.setText(
            format_scalar(nanometer.convert(scalars.nearest_neighbor_maximum_m), precision=4)
        )

        i0_unit = 'photons' if scalars.i0_is_calibrated else 'counts'
        dialog.i0_group.setTitle(f'I0 [{i0_unit}]')
        dialog.i0_mean_label.setText(format_scalar(scalars.i0_mean, precision=4))
        dialog.i0_std_label.setText(format_scalar(scalars.i0_std, precision=4))
        dialog.i0_minimum_label.setText(format_scalar(scalars.i0_minimum, precision=4))
        dialog.i0_maximum_label.setText(format_scalar(scalars.i0_maximum, precision=4))

        frame = ' (exit frame)' if scalars.is_exit_frame else ''
        positions = report.positions
        x_um = positions.get_coordinates_x_m() / micrometer.meters_per_unit
        y_um = positions.get_coordinates_y_m() / micrometer.meters_per_unit

        if report.i0 is None:
            colors = positions.get_indexes().astype(float)
            color_label = 'Scan Index'
        else:
            colors = report.i0
            color_label = f'I0 [{i0_unit}]'

        figure = dialog.probe_positions_plot_view.figure
        figure.clear()
        positions_axes = figure.add_subplot(111)
        scatter = positions_axes.scatter(x_um, y_um, c=colors, s=12, cmap='viridis')
        figure.colorbar(scatter, ax=positions_axes, label=color_label)
        positions_axes.invert_yaxis()
        positions_axes.set_aspect('equal')
        positions_axes.grid(True)
        positions_axes.set_xlabel(f'X{frame} [µm]')
        positions_axes.set_ylabel(f'Y{frame} [µm]')
        figure.tight_layout()
        dialog.probe_positions_plot_view.figure_canvas.draw()

        dialog.step_histogram_axes.clear()
        plot_histogram(
            dialog.step_histogram_axes,
            report.nearest_neighbor_distance_m / nanometer.meters_per_unit,
        )
        dialog.step_histogram_axes.set_xlabel(f'Nearest-Neighbor Distance{frame} [nm]')
        dialog.step_histogram_axes.set_ylabel('Positions')

        dialog.i0_axes.clear()
        dialog.i0_axes.set_visible(report.i0 is not None)

        if report.i0 is not None:
            dialog.i0_axes.plot(positions.get_indexes(), report.i0, '.', markersize=3)
            dialog.i0_axes.set_xlabel('Scan Index')
            dialog.i0_axes.set_ylabel(f'I0 [{i0_unit}]')
            dialog.i0_axes.grid(True)

        dialog.plot_view.figure.tight_layout()
        dialog.plot_view.figure_canvas.draw()
