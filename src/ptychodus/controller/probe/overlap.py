import logging
import math

import numpy

from ptychodus.api.constants import LengthUnit

from ...model.analysis import ProbeOverlapAnalyzer, ProbeOverlapMetrics
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbeOverlapDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

logger = logging.getLogger(__name__)

_SAVE_FILE_FILTER = 'NumPy Zipped Archive (*.npz)'
_PLACEHOLDER = '—'


def _format_scalar(value: float, *, suffix: str = '') -> str:
    if math.isnan(value):
        return _PLACEHOLDER
    if math.isinf(value):
        return f'∞{suffix}'
    return f'{value:.3g}{suffix}'


class ProbeOverlapViewController:
    def __init__(
        self,
        analyzer: ProbeOverlapAnalyzer,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._file_dialog_factory = file_dialog_factory
        self._metrics: ProbeOverlapMetrics | None = None
        self._dialog = ProbeOverlapDialog()

        self._visualization_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.visualization_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._visualization_parameters_controller = VisualizationParametersController(
            engine, self._dialog.visualization_parameters_view
        )
        self._dialog.save_button.clicked.connect(self._save_data)

    def analyze(self, product_index: int) -> None:
        self._metrics = None
        self._sync_dialog()

        try:
            product_name = self._analyzer.get_product_name(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Probe Overlap Analyzer', err)
            return

        self._dialog.setWindowTitle(f'Probe Overlap: {product_name}')
        self._dialog.open()

        try:
            self._metrics = self._analyzer.analyze(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Probe Overlap Analyzer', err)

        self._sync_dialog()

    def _sync_dialog(self) -> None:
        metrics = self._metrics
        view = self._dialog.metrics_view

        if metrics is None:
            self._visualization_widget_controller.clear_array()

            for label in (
                view.areal_redundancy_label,
                view.equivalent_linear_overlap_label,
                view.effective_probe_diameter_label,
                view.effective_step_size_label,
                view.mean_pairwise_overlap_label,
                view.median_pairwise_overlap_label,
                view.minimum_pairwise_overlap_label,
                view.maximum_pairwise_overlap_label,
                view.num_positions_label,
            ):
                label.setText(_PLACEHOLDER)

            return

        # NaN marks pixels no probe reaches; zero probes is the honest display value,
        # and the renderer warns about non-finite input on every repaint otherwise.
        self._visualization_widget_controller.set_array(
            numpy.nan_to_num(metrics.redundancy, nan=0.0), metrics.pixel_geometry
        )

        view.areal_redundancy_label.setText(_format_scalar(metrics.areal_redundancy))
        view.equivalent_linear_overlap_label.setText(
            _format_scalar(100.0 * metrics.equivalent_linear_overlap)
        )
        view.effective_probe_diameter_label.setText(
            _format_scalar(LengthUnit.NANOMETER.convert(metrics.effective_probe_diameter_m))
        )
        view.effective_step_size_label.setText(
            _format_scalar(LengthUnit.NANOMETER.convert(metrics.effective_step_size_m))
        )
        view.mean_pairwise_overlap_label.setText(_format_scalar(metrics.mean_pairwise_overlap))
        view.median_pairwise_overlap_label.setText(_format_scalar(metrics.median_pairwise_overlap))
        view.minimum_pairwise_overlap_label.setText(
            _format_scalar(metrics.minimum_pairwise_overlap)
        )
        view.maximum_pairwise_overlap_label.setText(
            _format_scalar(metrics.maximum_pairwise_overlap)
        )
        view.num_positions_label.setText(str(metrics.num_positions))

    def _save_data(self) -> None:
        if self._metrics is None:
            logger.warning('No probe overlap metrics to save!')
            return

        title = 'Save Probe Overlap Metrics'
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=[_SAVE_FILE_FILTER],
            selected_name_filter=_SAVE_FILE_FILTER,
        )

        if file_path:
            try:
                self._metrics.save_npz(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)
