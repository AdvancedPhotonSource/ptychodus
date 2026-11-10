from __future__ import annotations
import logging
import math

import numpy

from PyQt5.QtCore import QAbstractItemModel

from ptychodus.api.constants import LengthUnit
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import ProbeSizeMetrics, estimate_probe_size
from ptychodus.api.typing import RealArrayType

from ...model.analysis import ProbeAnalyzer, ProbeMetricsReport
from ...view.probe import ProbeMetricsDialog
from ...view.widgets import ExceptionDialog
from ..dashboard import format_scalar, wait_cursor
from ..data import FileDialogFactory
from ..rendering import VisualizationEngine
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


def compute_xy_metrics(
    intensity: RealArrayType, pixel_geometry: PixelGeometry
) -> ProbeSizeMetrics | None:
    try:
        return estimate_probe_size(intensity, pixel_geometry)
    except Exception:
        logger.exception('Failed to estimate probe size from XY intensity!')
        return None


def _format_length_nm(value_m: float) -> str:
    return format_scalar(LengthUnit.NANOMETER.convert(value_m), precision=4)


class ProbeMetricsViewController:
    def __init__(
        self,
        analyzer: ProbeAnalyzer,
        tree_model: QAbstractItemModel,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._analyzer = analyzer
        self._dialog = ProbeMetricsDialog()
        self._dialog.setWindowTitle('Probe Metrics')
        self._dialog.product_combo_box.setModel(tree_model)
        self._dialog.product_combo_box.activated.connect(self._analyze_current)

        modes_engine = VisualizationEngine(is_complex=True)
        self._modes_controller = VisualizationWidgetController(
            modes_engine, self._dialog.modes_widget, self._dialog.status_bar, file_dialog_factory
        )
        self._modes_parameters_controller = VisualizationParametersController(
            modes_engine, self._dialog.visualization_parameters_view
        )
        self._redundancy_controller = VisualizationWidgetController(
            VisualizationEngine(is_complex=False),
            self._dialog.redundancy_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )

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
            ExceptionDialog.show_exception('Probe Metrics', err)
            return

        self._show(report)

    def _clear(self) -> None:
        dialog = self._dialog

        for group in (dialog.size_group, dialog.entropy_group, dialog.modes_group):
            group.reset()

        dialog.overlap_group.reset()
        self._modes_controller.clear_array()
        self._redundancy_controller.clear_array()
        dialog.spectrum_plot_view.axes.clear()
        dialog.spectrum_plot_view.figure_canvas.draw()

    def _show(self, report: ProbeMetricsReport) -> None:
        dialog = self._dialog
        profile = report.scalars.profile
        size = profile.size

        if size is None:
            dialog.size_group.reset()
        else:
            dialog.major_axis_tilt_label.setText(
                format_scalar(math.degrees(size.major_axis_tilt_rad), precision=4)
            )
            dialog.fwhm_major_label.setText(_format_length_nm(size.fwhm_major_axis_length_m))
            dialog.fwhm_minor_label.setText(_format_length_nm(size.fwhm_minor_axis_length_m))
            dialog.rms_major_label.setText(_format_length_nm(size.rms_major_axis_length_m))
            dialog.rms_minor_label.setText(_format_length_nm(size.rms_minor_axis_length_m))
            dialog.encircled_energy_label.setText(
                _format_length_nm(size.encircled_energy_diameter_m)
            )

        dialog.real_space_entropy_label.setText(
            format_scalar(profile.entropy.real_space_intensity_entropy, precision=4)
        )
        dialog.spectral_entropy_label.setText(
            format_scalar(profile.entropy.spectral_entropy, precision=4)
        )

        dialog.modes_group.setTitle(
            'Modes (OPR Mean Field)' if report.scalars.has_opr_weights else 'Modes'
        )
        dialog.num_modes_label.setText(str(profile.num_modes))
        dialog.purity_label.setText(format_scalar(profile.purity, precision=4))
        dialog.von_neumann_entropy_label.setText(
            format_scalar(profile.von_neumann_entropy, precision=4)
        )
        dialog.effective_mode_count_label.setText(
            format_scalar(profile.effective_mode_count, precision=4)
        )

        overlap = report.scalars.overlap

        if overlap is None:
            dialog.overlap_group.reset()
        else:
            dialog.areal_redundancy_label.setText(format_scalar(overlap.areal_redundancy))
            dialog.equivalent_linear_overlap_label.setText(
                format_scalar(100.0 * overlap.equivalent_linear_overlap)
            )
            dialog.effective_probe_diameter_label.setText(
                _format_length_nm(overlap.effective_probe_diameter_m)
            )
            dialog.effective_step_size_label.setText(
                _format_length_nm(overlap.effective_step_size_m)
            )
            dialog.mean_pairwise_overlap_label.setText(format_scalar(overlap.mean_pairwise_overlap))
            dialog.median_pairwise_overlap_label.setText(
                format_scalar(overlap.median_pairwise_overlap)
            )
            dialog.minimum_pairwise_overlap_label.setText(
                format_scalar(overlap.minimum_pairwise_overlap)
            )
            dialog.maximum_pairwise_overlap_label.setText(
                format_scalar(overlap.maximum_pairwise_overlap)
            )
            dialog.num_positions_label.setText(str(overlap.num_positions))

        orthogonal_probe = report.profile.orthogonal_probe
        self._modes_controller.set_array(
            orthogonal_probe.get_incoherent_modes_flattened(),
            orthogonal_probe.get_pixel_geometry(),
        )
        self._modes_controller.zoom_to_fit()

        if report.overlap is None:
            self._redundancy_controller.clear_array()
        else:
            # NaN marks pixels no probe reaches; zero probes is the honest display value.
            self._redundancy_controller.set_array(
                numpy.nan_to_num(report.overlap.redundancy, nan=0.0),
                report.overlap.pixel_geometry,
            )
            self._redundancy_controller.zoom_to_fit()

        axes = dialog.spectrum_plot_view.axes
        axes.clear()
        relative_power = report.profile.relative_power
        axes.bar(numpy.arange(relative_power.size), 100.0 * relative_power)
        axes.set_xlabel('Orthogonal Mode')
        axes.set_ylabel('Power [%]')
        axes.set_xticks(numpy.arange(relative_power.size))
        axes.grid(True, axis='y')
        dialog.spectrum_plot_view.figure.tight_layout()
        dialog.spectrum_plot_view.figure_canvas.draw()
