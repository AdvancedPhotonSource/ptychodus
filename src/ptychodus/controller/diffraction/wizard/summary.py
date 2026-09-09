from __future__ import annotations

from collections.abc import Callable
import logging

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import (
    QGraphicsRectItem,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ptychodus.api.diffraction import CropRegion, DiffractionPattern
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.preprocess.diffraction import FilterValuesStep

from ....model.diffraction import (
    DiffractionSettings,
    DiffractionSummaryService,
    PrepPipelineBuilder,
)
from ....model.visualization import VisualizationEngine
from ....view.image import ImageView
from ....view.widgets import ExceptionDialog, TaskStatusView

from ...data import FileDialogFactory
from ...image import ImageController
from ...task_status import TaskStatusController
from ..detector_extent import DetectorExtentSource

logger = logging.getLogger(__name__)


class SummaryPanelViewController(Observer):
    """Collapsible right pane of the Processing wizard page.

    The pane is hidden on arrival and is revealed by a checkable toggle button
    that the Processing page places at the bottom of its left column (see
    :meth:`get_toggle_widget`). Expanding kicks off a summarize when none is
    cached for the pending dataset, so the pane is never shown empty; a compute
    that fails to dispatch springs the toggle back to collapsed. Collapsing
    returns the full page width to the preprocess-pipeline group boxes.

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

    The displayed frame is a live preview: the value filter from the current
    :class:`DiffractionPrepPlan` is applied to the cached mean_pattern on every
    settings change. Only that one step is previewed. Binning, upsample,
    padding, and the flip/transpose steps change the frame's extent and
    orientation, so applying them here would stop the pane reading as a view of
    the detector; crop is likewise shown as an overlay rather than applied. The
    frame therefore stays at full detector extent in raw detector coordinates,
    which is also what lets the overlay and the pixel geometry stay untransformed.

    Two summary-driven actions — Estimate Beam Center and Robust Statistics —
    live on the Crop and Total Counts Filter group boxes respectively (see
    :class:`CropViewController` and :class:`RobustStatisticsDialog` in
    processing.py). They read `summary_service.get_last_summary()` on click
    and observe the same task_monitor for their enable state; both keep
    working while this pane is collapsed.
    """

    def __init__(
        self,
        diffraction_settings: DiffractionSettings,
        summary_service: DiffractionSummaryService,
        summary_visualization_engine: VisualizationEngine,
        extent_source: DetectorExtentSource,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
        get_pending_dataset_index: Callable[[], int],
    ) -> None:
        super().__init__()
        self._diffraction_settings = diffraction_settings
        self._summary_service = summary_service
        self._extent_source = extent_source
        self._get_pending_dataset_index = get_pending_dataset_index
        self._last_rendered_run_id = summary_service.get_last_run_id()
        self._last_rendered_counts_run_id = summary_service.get_last_total_counts_run_id()
        self._summarized_dataset_index = -1
        # Stateless; reads live settings on each get_plan call, so one instance
        # serves the pane's lifetime.
        self._pipeline_builder = PrepPipelineBuilder(diffraction_settings)
        self._raw_mean_pattern: DiffractionPattern | None = None
        self._raw_pixel_geometry: PixelGeometry | None = None

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

        # --- Task status (bottom) ---
        self._task_status_view = TaskStatusView()
        self._task_status_controller = TaskStatusController(
            summary_service.task_monitor, self._task_status_view
        )

        # --- Collapse toggle, hosted by the Processing page's left column ---
        self._toggle_button = QToolButton()
        self._toggle_button.setText('Summary')
        self._toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle_button.setToolTip('Summarize the pending dataset and show the summary pane.')
        self._toggle_button.toggled.connect(self._on_toggled)

        # --- Assembly ---
        # Only the image and the plot share the vertical splitter; the status
        # row is fixed below it so it cannot be dragged away.
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.addWidget(self._image_view)
        self._splitter.addWidget(plot_widget)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 1)

        panel_layout = QVBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        panel_layout.addWidget(self._splitter)
        panel_layout.addWidget(self._task_status_view)

        self._panel = QWidget()
        self._panel.setLayout(panel_layout)
        self._panel.setVisible(False)

        # DiffractionSettings re-broadcasts its whole parameter group, so one
        # observation covers every control the preview and the overlays read.
        diffraction_settings.add_observer(self)
        extent_source.add_observer(self)
        summary_service.task_monitor.add_observer(self)

        self._update_threshold_overlay()

    def get_widget(self) -> QWidget:
        return self._panel

    def get_toggle_widget(self) -> QWidget:
        """Checkable control that shows and hides :meth:`get_widget`.

        Returned separately so the Processing page can seat it in the left
        column, where it stays visible while the pane itself is collapsed.
        """
        return self._toggle_button

    # ------------------------------------------------------------------
    # Expand / collapse

    def _on_toggled(self, checked: bool) -> None:
        if checked:
            if self._needs_compute() and not self._start_compute():
                # Re-enters this handler with checked=False, which collapses.
                self._toggle_button.setChecked(False)
                return

            self._toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self._panel.setVisible(True)
            self._apply_minimum_width()
            # Refreshes settings changes that were skipped while hidden.
            self._render_preview()
        else:
            # Read the width before hiding; a hidden widget reports none.
            expanded_width = self._panel.width()
            self._toggle_button.setArrowType(Qt.ArrowType.RightArrow)
            self._panel.setVisible(False)
            self._shrink_window_by(expanded_width)

    def _apply_minimum_width(self) -> None:
        """Pin the pane to its preferred width, once it has real geometry.

        Deferred to the first expand so the size hint reflects a laid-out,
        style-polished widget rather than one that has never been shown.
        """
        if self._panel.minimumWidth() == 0:
            self._panel.setMinimumWidth(self._panel.sizeHint().width())

    def _shrink_window_by(self, width: int) -> None:
        """Give the width the pane occupied back to the window.

        Showing the pane raises the page's minimum width and Qt grows the
        wizard to honor it, but hiding it again never shrinks the wizard back.
        """
        window = self._panel.window()
        target = max(window.minimumSizeHint().width(), window.width() - width)
        window.resize(target, window.height())

    # ------------------------------------------------------------------
    # Compute Summary

    def _needs_compute(self) -> bool:
        return (
            self._summary_service.get_last_summary() is None
            or self._summarized_dataset_index != self._get_pending_dataset_index()
        )

    def _start_compute(self) -> bool:
        """Dispatch a summarize; return whether the pane has data to show."""
        if self._summary_service.task_monitor.is_processing:
            return True

        dataset_index = self._get_pending_dataset_index()

        try:
            self._summary_service.compute(dataset_index)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Compute Summary', exc)
            return False

        self._summarized_dataset_index = dataset_index
        return True

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

        counts_run_id = self._summary_service.get_last_total_counts_run_id()
        if counts_run_id != self._last_rendered_counts_run_id:
            self._last_rendered_counts_run_id = counts_run_id
            self._render_counts()

    # ------------------------------------------------------------------
    # Rendering

    def _render_summary(self) -> None:
        summary = self._summary_service.get_last_summary()

        if summary is None:
            self._raw_mean_pattern = None
            self._raw_pixel_geometry = None
            self._image_controller.clear_array()
        else:
            dataset_index = self._get_pending_dataset_index()
            repository = self._summary_service._api.get_repository()  # noqa: SLF001

            if 0 <= dataset_index < len(repository):
                # Cached so a settings change can redraw the preview without
                # touching the repository again.
                self._raw_mean_pattern = summary.mean_pattern
                self._raw_pixel_geometry = repository[dataset_index].get_raw_pixel_geometry()
            else:
                self._raw_mean_pattern = None
                self._raw_pixel_geometry = None

        self._render_counts()
        self._render_preview()
        self._update_threshold_overlay()

    def _render_counts(self) -> None:
        """Plot per-pattern total counts, preferring the measured post-pipeline pass.

        The threshold lines drawn over this axes are the filter's bounds, and the
        filter compares them against post-crop, post-pipeline counts. So whenever
        a counts pass has run, its numbers are what belong here; the summary's raw
        full-detector totals are only a stand-in until then, and the axes label
        says which of the two is showing.
        """
        measured = self._summary_service.get_last_total_counts()

        if measured is not None:
            self._counts_line.set_data(measured.indexes, measured.total_counts)
            self._axes.set_ylabel('Total Counts (processed)')
        else:
            summary = self._summary_service.get_last_summary()

            if summary is None:
                self._counts_line.set_data([], [])
                return

            self._counts_line.set_data(summary.indexes, summary.total_counts)
            self._axes.set_ylabel('Total Counts (raw)')

        self._axes.relim()
        self._axes.autoscale_view()

    def _render_preview(self) -> None:
        """Redraw the mean pattern with the previewable processing controls applied.

        Only the value filter is applied; see the class docstring for why the
        geometry-changing steps and the crop are not. Two fidelity limits are
        inherent to previewing on a mean frame and cannot be fixed here:

        - Thresholding the mean is not thresholding each frame and averaging. A
          pixel whose per-frame values straddle the upper bound is partially
          zeroed in production but survives whole here when its mean sits inside
          the bounds. This shows which detector regions the filter targets, not
          the exact result.
        - ``summarize_dataset`` inpaints bad pixels into mean_pattern, whereas
          production zeroes them before the pipeline, so the filter sees
          interpolated values where production sees zeros.
        """
        if self._raw_mean_pattern is None or self._raw_pixel_geometry is None:
            # No frame behind it, so the rectangle would mark nothing.
            self._crop_overlay.hide()
            return

        if not self._panel.isVisible():
            # Settings edits while collapsed cost nothing; _on_toggled refreshes.
            return

        try:
            plan = self._pipeline_builder.get_plan(self._extent_source.get_extent())
        except ValueError:
            # Transient while the user types a bin size that does not divide the
            # crop extent. Binning is not previewed, but get_plan still validates
            # it. Hold the last good frame -- a modal dialog per keystroke would
            # make the spin boxes unusable.
            logger.debug('Skipping preview refresh; incomplete prep plan.', exc_info=True)
            return

        frame = self._raw_mean_pattern
        step = next((s for s in plan.pipeline.steps if isinstance(s, FilterValuesStep)), None)

        if step is not None:
            frame = step.apply(frame)

        self._image_controller.set_array(frame, self._raw_pixel_geometry)
        self._update_crop_overlay(plan.read_region)

    # ------------------------------------------------------------------
    # Overlay updates

    def _update_crop_overlay(self, read_region: CropRegion | None) -> None:
        """Draw the crop rectangle the prep plan actually resolved.

        Taking the region from the plan rather than re-deriving it from the
        settings picks up ``clamp_to_detector_extent``, so the rectangle cannot
        stray outside the detector the way a hand-rolled center/extent
        calculation can.
        """
        if read_region is None:
            self._crop_overlay.hide()
            return

        self._crop_overlay.setRect(
            QRectF(
                float(read_region.x_range[0]),
                float(read_region.y_range[0]),
                float(read_region.width_px),
                float(read_region.height_px),
            )
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
        elif observable is self._diffraction_settings or observable is self._extent_source:
            self._render_preview()
            self._update_threshold_overlay()
