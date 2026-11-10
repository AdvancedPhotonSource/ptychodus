from __future__ import annotations
import logging

import numpy

from PyQt5.QtCore import QAbstractItemModel

from ptychodus.api.constants import PI, TWO_PI, LengthUnit

from ...model.analysis import ProbePositionsAnalyzer, ProbePositionsComparisonReport
from ...view.probe_positions import ProbePositionsCompareDialog
from ...view.widgets import ExceptionDialog
from ..dashboard import format_scalar, plot_histogram, select_pair, wait_cursor

logger = logging.getLogger(__name__)


class ProbePositionsCompareViewController:
    def __init__(self, analyzer: ProbePositionsAnalyzer, table_model: QAbstractItemModel) -> None:
        self._analyzer = analyzer
        self._dialog = ProbePositionsCompareDialog()
        self._dialog.setWindowTitle('Compare Probe Positions')
        self._dialog.reference_combo_box.setModel(table_model)
        self._dialog.reference_combo_box.activated.connect(self._compare_current)
        self._dialog.test_combo_box.setModel(table_model)
        self._dialog.test_combo_box.activated.connect(self._compare_current)

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
            ExceptionDialog.show_exception('Compare Probe Positions', err)
            return

        self._show(report)

    def _clear(self) -> None:
        dialog = self._dialog
        dialog.matching_group.reset()
        dialog.displacement_group.reset()
        dialog.field_plot_view.figure.clear()
        dialog.field_plot_view.figure_canvas.draw()

        for axes in (
            dialog.magnitude_histogram_axes,
            dialog.direction_histogram_axes,
            dialog.index_axes,
        ):
            axes.clear()

        dialog.plot_view.figure_canvas.draw()

    def _show(self, report: ProbePositionsComparisonReport) -> None:
        dialog = self._dialog
        scalars = report.scalars
        nanometer = LengthUnit.NANOMETER
        micrometer = LengthUnit.MICROMETER

        dialog.num_matched_label.setText(str(scalars.num_matched))
        dialog.num_reference_only_label.setText(str(scalars.num_reference_only))
        dialog.num_test_only_label.setText(str(scalars.num_test_only))
        dialog.mean_x_label.setText(
            format_scalar(nanometer.convert(scalars.mean_displacement_x_m), precision=4)
        )
        dialog.mean_y_label.setText(
            format_scalar(nanometer.convert(scalars.mean_displacement_y_m), precision=4)
        )
        dialog.rms_label.setText(
            format_scalar(nanometer.convert(scalars.rms_magnitude_m), precision=4)
        )
        dialog.median_label.setText(
            format_scalar(nanometer.convert(scalars.median_magnitude_m), precision=4)
        )
        dialog.maximum_label.setText(
            format_scalar(nanometer.convert(scalars.maximum_magnitude_m), precision=4)
        )

        frame = ' (exit frame)' if scalars.is_exit_frame else ''
        displacement_nm = report.displacement_xy_m / nanometer.meters_per_unit
        magnitude_nm = report.magnitude_m / nanometer.meters_per_unit
        reference_xy_um = report.reference_xy_m / micrometer.meters_per_unit

        figure = dialog.field_plot_view.figure
        figure.clear()
        field_axes = figure.add_subplot(111)
        quiver = field_axes.quiver(
            reference_xy_um[:, 0],
            reference_xy_um[:, 1],
            displacement_nm[:, 0],
            displacement_nm[:, 1],
            magnitude_nm,
            angles='xy',
            cmap='viridis',
        )
        figure.colorbar(quiver, ax=field_axes, label='Displacement [nm]')
        field_axes.invert_yaxis()
        field_axes.set_aspect('equal')
        field_axes.grid(True)
        field_axes.set_xlabel(f'X{frame} [µm]')
        field_axes.set_ylabel(f'Y{frame} [µm]')
        field_axes.set_title('Arrow lengths are scaled for visibility', fontsize='small')
        figure.tight_layout()
        dialog.field_plot_view.figure_canvas.draw()

        dialog.magnitude_histogram_axes.clear()
        plot_histogram(dialog.magnitude_histogram_axes, magnitude_nm)
        dialog.magnitude_histogram_axes.set_xlabel('Displacement [nm]')
        dialog.magnitude_histogram_axes.set_ylabel('Positions')

        num_direction_bins = 36
        counts, edges = numpy.histogram(
            report.direction_rad, bins=num_direction_bins, range=(-PI, PI)
        )
        dialog.direction_histogram_axes.clear()
        dialog.direction_histogram_axes.bar(
            edges[:-1], counts, width=TWO_PI / num_direction_bins, align='edge'
        )
        dialog.direction_histogram_axes.set_title('Direction', fontsize='small')

        dialog.index_axes.clear()
        dialog.index_axes.plot(report.indexes, displacement_nm[:, 0], '.', markersize=3, label='dx')
        dialog.index_axes.plot(report.indexes, displacement_nm[:, 1], '.', markersize=3, label='dy')
        dialog.index_axes.set_xlabel('Scan Index')
        dialog.index_axes.set_ylabel('Displacement [nm]')
        dialog.index_axes.grid(True)
        dialog.index_axes.legend(loc='best', fontsize='small')

        dialog.plot_view.figure.tight_layout()
        dialog.plot_view.figure_canvas.draw()
