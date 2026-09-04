from __future__ import annotations

from collections.abc import Callable
import logging

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import (
    QGraphicsRectItem,
    QGridLayout,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ptychodus.api.observer import Observable, Observer

from ....model.diffraction import DiffractionSettings, DiffractionSummaryService
from ....model.visualization import VisualizationEngine
from ....view.image import ImageView
from ....view.widgets import ExceptionDialog, TaskStatusView

from ...data import FileDialogFactory
from ...image import ImageController
from ...task_status import TaskStatusController

logger = logging.getLogger(__name__)


class SummaryPanelViewController(Observer):
    """Right pane of the Processing wizard page.

    Delegates the summarize compute to :class:`DiffractionSummaryService` — the
    controller never touches :class:`TaskManager`. Observes the service's
    :class:`DiffractionSummaryTaskMonitor` for state transitions and pulls a
    fresh :class:`DiffractionSummary` when the service's run-id advances.

    - mean_pattern renders in an :class:`ImageView` backed by its own dedicated
      :class:`VisualizationEngine` (``ModelCore.summary_visualization_engine``,
      distinct from ``pattern_visualization_engine`` used by the main diffraction
      pane) with a live crop-rectangle overlay driven by the crop settings. The
      two engines are kept separate so autoscaling the mean_pattern's color
      range does not disturb whatever range the user set on the main pane, and
      vice versa.
    - total_counts is plotted on a log-y matplotlib axes with live threshold
      lines driven by the total-counts settings.

    Two summary-driven actions — Estimate Beam Center and Robust Statistics —
    live on the Crop and Total Counts Filter group boxes respectively (see
    :class:`CropViewController` and :class:`RobustStatisticsDialog` in
    processing.py). They read `summary_service.get_last_summary()` on click
    and observe the same task_monitor for their enable state.
    """

    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        summary_service: DiffractionSummaryService,
        summary_visualization_engine: VisualizationEngine,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
        get_pending_dataset_index: Callable[[], int],
    ) -> None:
        super().__init__()
        self._diffraction_settings = diffraction_settings
        self._summary_service = summary_service
        self._get_pending_dataset_index = get_pending_dataset_index
        self._last_rendered_run_id = summary_service.get_last_run_id()

        # --- mean_pattern image view (top) ---
        self._image_view = ImageView()
        self._image_controller = ImageController(
            summary_visualization_engine,
            self._image_view,
            status_bar,
            file_dialog_factory,
        )
        crop_pen = QPen(Qt.PenStyle.DashLine)
        crop_pen.setColor(QColor(Qt.GlobalColor.yellow))
        crop_pen.setWidth(2)
        crop_pen.setCosmetic(True)
        self._crop_overlay = QGraphicsRectItem(self._image_controller.get_item())
        self._crop_overlay.setPen(crop_pen)
        self._crop_overlay.setZValue(100)
        self._crop_overlay.hide()

        # --- total_counts plot (middle) ---
        self._figure = Figure()
        self._figure_canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        self._axes.set_yscale('log')
        self._axes.set_xlabel('Scan Index')
        self._axes.set_ylabel('Total Counts')
        self._axes.grid(True, which='both', linestyle=':', alpha=0.5)
        (self._counts_line,) = self._axes.plot([], [], '.', markersize=2)
        self._lower_line = self._axes.axhline(
            1.0, visible=False, color='tab:red', linestyle='--', linewidth=1
        )
        self._upper_line = self._axes.axhline(
            1.0, visible=False, color='tab:red', linestyle='--', linewidth=1
        )
        plot_widget = QWidget()
        navigation_toolbar = NavigationToolbar(self._figure_canvas, plot_widget)
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(navigation_toolbar)
        plot_layout.addWidget(self._figure_canvas)
        plot_widget.setLayout(plot_layout)

        # --- Summary controls (bottom) ---
        self._compute_button = QPushButton('Compute Summary')
        self._compute_button.clicked.connect(self._on_compute_clicked)
        self._task_status_view = TaskStatusView()
        self._task_status_controller = TaskStatusController(
            summary_service.task_monitor, self._task_status_view
        )

        controls_layout = QGridLayout()
        controls_layout.addWidget(self._compute_button, 0, 0)
        controls_layout.addWidget(self._task_status_view, 1, 0)
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)

        # --- Assembly ---
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self._image_view)
        self._splitter.addWidget(plot_widget)
        self._splitter.addWidget(controls_widget)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setStretchFactor(2, 0)

        self._observed_settings: tuple[Observable, ...] = (
            diffraction_settings.crop_enabled,
            diffraction_settings.beam_center_x_px,
            diffraction_settings.beam_center_y_px,
            diffraction_settings.crop_width_px,
            diffraction_settings.crop_height_px,
            diffraction_settings.total_counts_lower_bound_enabled,
            diffraction_settings.total_counts_lower_bound,
            diffraction_settings.total_counts_upper_bound_enabled,
            diffraction_settings.total_counts_upper_bound,
        )
        for parameter in self._observed_settings:
            parameter.add_observer(self)
        summary_service.task_monitor.add_observer(self)

        self._update_crop_overlay()
        self._update_threshold_overlay()
        self._sync_button_states()

    def get_widget(self) -> QWidget:
        return self._splitter

    # ------------------------------------------------------------------
    # Compute Summary

    def _on_compute_clicked(self) -> None:
        if self._summary_service.task_monitor.is_processing:
            return
        try:
            self._summary_service.compute(self._get_pending_dataset_index())
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Compute Summary', exc)

    def _sync_button_states(self) -> None:
        is_processing = self._summary_service.task_monitor.is_processing
        self._compute_button.setEnabled(not is_processing)

    def _handle_task_monitor_notification(self) -> None:
        monitor = self._summary_service.task_monitor
        error = monitor.get_last_error()

        if isinstance(error, Exception) and not monitor.is_processing:
            # The monitor holds the last-run error until the next run enters,
            # and only fires this handler again at __enter__/update/__exit__.
            # __enter__ clears the error, so this reports at most once per run.
            logger.exception(error)
            ExceptionDialog.show_exception('Compute Summary', error)

        run_id = self._summary_service.get_last_run_id()
        if run_id != self._last_rendered_run_id:
            self._last_rendered_run_id = run_id
            self._render_summary()

        self._sync_button_states()

    # ------------------------------------------------------------------
    # Rendering

    def _render_summary(self) -> None:
        summary = self._summary_service.get_last_summary()

        if summary is None:
            self._image_controller.clear_array()
            self._counts_line.set_data([], [])
        else:
            dataset_index = self._get_pending_dataset_index()
            repository = self._summary_service._api.get_repository()  # noqa: SLF001

            if 0 <= dataset_index < len(repository):
                pixel_geometry = repository[dataset_index].get_raw_pixel_geometry()
                self._image_controller.set_array(summary.mean_pattern, pixel_geometry)

            self._counts_line.set_data(summary.indexes, summary.total_counts)
            self._axes.relim()
            self._axes.autoscale_view()

        self._update_crop_overlay()
        self._update_threshold_overlay()

    # ------------------------------------------------------------------
    # Overlay updates

    def _update_crop_overlay(self) -> None:
        if not self._diffraction_settings.crop_enabled.get_value():
            self._crop_overlay.hide()
            return

        width = self._diffraction_settings.crop_width_px.get_value()
        height = self._diffraction_settings.crop_height_px.get_value()
        x_center = self._diffraction_settings.beam_center_x_px.get_value()
        y_center = self._diffraction_settings.beam_center_y_px.get_value()
        x_start = x_center - width // 2
        y_start = y_center - height // 2
        self._crop_overlay.setRect(
            QRectF(float(x_start), float(y_start), float(width), float(height))
        )
        self._crop_overlay.show()

    def _update_threshold_overlay(self) -> None:
        lower_enabled = self._diffraction_settings.total_counts_lower_bound_enabled.get_value()
        upper_enabled = self._diffraction_settings.total_counts_upper_bound_enabled.get_value()
        lower_value = self._diffraction_settings.total_counts_lower_bound.get_value()
        upper_value = self._diffraction_settings.total_counts_upper_bound.get_value()

        # Log-scale y clamps at a positive floor; a lower bound of 0 would be -inf.
        lower_y = max(1.0, float(lower_value))
        upper_y = max(1.0, float(upper_value))
        self._lower_line.set_ydata([lower_y, lower_y])
        self._lower_line.set_visible(lower_enabled)
        self._upper_line.set_ydata([upper_y, upper_y])
        self._upper_line.set_visible(upper_enabled)
        self._figure_canvas.draw_idle()

    def _update(self, observable: Observable) -> None:
        if observable is self._summary_service.task_monitor:
            self._handle_task_monitor_notification()
        elif observable in self._observed_settings:
            self._update_crop_overlay()
            self._update_threshold_overlay()
