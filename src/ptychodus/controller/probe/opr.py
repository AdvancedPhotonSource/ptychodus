from __future__ import annotations
from enum import Enum, auto
import logging

import numpy

from matplotlib.axes import Axes
from matplotlib.backend_bases import DrawEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.lines import Line2D

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QListWidgetItem,
    QMessageBox,
    QSpinBox,
    QStyle,
    QToolButton,
    QWidget,
)

from ptychodus.api.probe import OPRModeSeries, ProbeSequence
from ptychodus.api.typing import ComplexArrayType, IntegerArrayType, RealArrayType

from ...model.analysis import OPRModeAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.probe import OPRModeDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController

logger = logging.getLogger(__name__)

_SAVE_FILE_FILTER = 'NumPy Zipped Archive (*.npz)'

# Positions sampled when fixing the color scale. A locked scale is what keeps playback
# from flickering, and sampling is what keeps locking it cheap on a long scan; a position
# outside the sample whose amplitude exceeds the others simply clips.
_NUM_DISPLAY_RANGE_SAMPLES = 64


class _ColorBy(Enum):
    """Quantity the scan grid colors each probe position by."""

    DEVIATION = auto()
    PRIMARY = auto()
    TOTAL = auto()
    MEASURED = auto()
    WEIGHT = auto()


def _subsample_indexes(num_positions: int, num_samples: int) -> IntegerArrayType:
    if num_positions <= num_samples:
        return numpy.arange(num_positions)

    return numpy.unique(numpy.linspace(0, num_positions - 1, num_samples).astype(int))


class _BlittedMarker:
    """One artist repainted over a cached bitmap of everything behind it.

    Re-rasterizing a plot of several thousand points costs an order of magnitude more
    than the probe image beside it, and during playback the only thing that moves is the
    marker. Caching the rest of the figure and restoring it keeps the per-frame cost
    independent of how many points the plot holds.

    The artist is marked animated, so a full draw leaves it out and the cache captured
    during that draw holds only the static content.
    """

    def __init__(self, canvas: FigureCanvasQTAgg) -> None:
        self._canvas = canvas
        self._axes: Axes | None = None
        self._artist: Line2D | None = None
        self._background: object | None = None
        canvas.mpl_connect('draw_event', self._handle_draw)

    def set_artist(self, axes: Axes | None, artist: Line2D | None) -> None:
        """Adopt the marker of a freshly rebuilt figure, or drop it with ``None``."""
        if artist is not None:
            artist.set_animated(True)

        self._axes = axes
        self._artist = artist
        self._background = None

    def _handle_draw(self, event: DrawEvent) -> None:
        # Any full draw invalidates the cache: a resize, a toolbar zoom, or a rebuild of
        # the figure. Recapture it, then put the animated artist back on top.
        self._background = self._canvas.copy_from_bbox(self._canvas.figure.bbox)
        self._draw_artist()

    def _draw_artist(self) -> None:
        if self._axes is not None and self._artist is not None:
            self._axes.draw_artist(self._artist)

    def update(self) -> None:
        """Repaint just the marker, or fall back to a full draw before the first one."""
        if self._background is None or self._artist is None:
            self._canvas.draw_idle()
            return

        self._canvas.restore_region(self._background)
        self._draw_artist()
        self._canvas.blit(self._canvas.figure.bbox)


class OPRModeViewController:
    """Drives the dialog showing how a probe's coherent (OPR) basis varies over a scan."""

    def __init__(
        self,
        analyzer: OPRModeAnalyzer,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
        parent: QWidget,
    ) -> None:
        self._analyzer = analyzer
        self._file_dialog_factory = file_dialog_factory
        self._parent = parent

        self._product_index = -1
        self._series: OPRModeSeries | None = None
        self._probes: ProbeSequence | None = None
        self._measured_photon_count: RealArrayType | None = None
        self._num_positions = 0
        self._was_playing_before_drag = False
        self._position_marker: Line2D | None = None
        self._grid_highlight: Line2D | None = None
        # Cached alongside the scatter so the playback path never goes back to the
        # repository for coordinates it has already drawn.
        self._grid_x_m: RealArrayType = numpy.empty(0)
        self._grid_y_m: RealArrayType = numpy.empty(0)

        # A plain spin box rather than a settings-backed one: the range is the probe's
        # coherent mode count, and IntegerParameter bounds are immutable.
        self._mode_spin_box = QSpinBox()
        self._mode_spin_box.setToolTip('Coherent (OPR) Mode')
        self._mode_spin_box.setRange(1, 1)
        self._mode_spin_box.setEnabled(False)

        style = parent.style()
        self._play_icon = style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay) if style else None
        self._pause_icon = (
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPause) if style else None
        )

        self._play_button = QToolButton()
        self._play_button.setCheckable(True)
        self._play_button.setToolTip('Play through the probe positions')
        # Text as well as the icon: a platform style that returns a null media pixmap
        # would otherwise leave a blank square.
        self._play_button.setText('Play')
        self._play_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._frame_rate_spin_box = QDoubleSpinBox()
        self._frame_rate_spin_box.setRange(0.1, 60.0)
        self._frame_rate_spin_box.setSingleStep(1.0)
        self._frame_rate_spin_box.setDecimals(1)
        self._frame_rate_spin_box.setValue(10.0)
        self._frame_rate_spin_box.setSuffix(' fps')
        self._frame_rate_spin_box.setToolTip(
            'Target frame rate. A frame that takes longer than this to render simply'
            ' lowers the rate rather than queueing up behind the next one.'
        )

        # Single-shot and re-armed once a frame has finished rendering: that is what makes
        # the rate best-effort, and it also keeps a frame whose canvas draw spins the event
        # loop from re-entering the advance.
        self._playback_timer = QTimer()
        self._playback_timer.setSingleShot(True)
        self._playback_timer.timeout.connect(self._advance_frame)

        self._dialog = OPRModeDialog(
            self._mode_spin_box,
            self._play_button,
            self._frame_rate_spin_box,
            parent,
        )
        self._image_controller = ImageController(
            engine,
            self._dialog.probe_view,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._engine = engine
        self._series_marker = _BlittedMarker(self._dialog.series_plot_view.figure_canvas)
        self._grid_marker = _BlittedMarker(self._dialog.scan_grid_plot_view.figure_canvas)

        self._dialog.position_slider.valueChanged.connect(self._update_current_position)
        # Dragging while playing would otherwise issue a render per drag tick and leave the
        # timer stepping away from wherever the user let go.
        self._dialog.position_slider.sliderPressed.connect(self._handle_slider_pressed)
        self._dialog.position_slider.sliderReleased.connect(self._handle_slider_released)
        self._dialog.save_button.clicked.connect(self._save_series)
        self._play_button.toggled.connect(self._handle_play_toggled)
        self._mode_spin_box.valueChanged.connect(self._update_image)
        self._dialog.curves_view.color_by_combo_box.currentIndexChanged.connect(
            self._redraw_scan_grid
        )
        self._dialog.curves_view.normalize_check_box.toggled.connect(self._redraw_series)
        self._dialog.curves_view.mode_list_widget.itemChanged.connect(self._redraw_series)

        for button in (
            self._dialog.display_view.composed_button,
            self._dialog.display_view.deviation_button,
            self._dialog.display_view.basis_button,
        ):
            button.toggled.connect(self._handle_display_toggled)

        self._dialog.display_view.composed_button.setChecked(True)
        # Connected once: the dialog is built here and re-opened per launch, so connecting
        # in launch would stack duplicates.
        self._dialog.finished.connect(self._handle_dialog_finished)

    # ------------------------------------------------------------------ launching

    def launch(self, product_index: int) -> None:
        self._stop_playback()
        self._product_index = product_index
        self._series = None
        self._probes = None
        self._measured_photon_count = None
        self._num_positions = 0

        try:
            probes = self._analyzer.get_probes(product_index)
            product_name = self._analyzer.get_product_name(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('OPR Mode Analyzer', err)
            return

        if probes.get_opr_weights_or_none() is None:
            QMessageBox.information(
                self._parent,
                'Coherent (OPR) Modes',
                f'This probe has {probes.num_coherent_modes} coherent (OPR) mode(s) and no'
                ' per-position weights, so there is nothing that varies across the scan to'
                ' show.\n\nWeights come from a reconstruction run with OPR enabled, or from'
                ' a product file that stored them.',
            )
            return

        self._dialog.setWindowTitle(f'Coherent (OPR) Modes: {product_name}')
        self._dialog.open()

        try:
            self._series = self._analyzer.analyze(product_index)
            self._probes = probes
            self._measured_photon_count = self._analyzer.get_measured_photon_counts(product_index)
            num_scan_points = len(self._analyzer.get_probe_positions(product_index))
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('OPR Mode Analyzer', err)
            self._series = None
            num_scan_points = 0

        if self._series is not None:
            self._num_positions = self._series.num_positions

            if self._num_positions != num_scan_points:
                self._dialog.status_bar.showMessage(
                    f'OPR weights describe {self._num_positions} probe position(s) but the'
                    f' product has {num_scan_points}; the scan grid shows the shorter of'
                    ' the two.'
                )

        self._sync_model_to_view()

    # ------------------------------------------------------------------ probe fields

    def _get_basis(self) -> ComplexArrayType:
        if self._probes is None:
            raise ValueError('No probe ensemble!')

        return self._probes.get_array()[:, 0, :, :]

    def _get_field(self, index: int) -> ComplexArrayType:
        """Complex field the image pane shows for probe position ``index``."""
        series = self._series

        if series is None:
            raise ValueError('No OPR mode series!')

        basis = self._get_basis()
        display_view = self._dialog.display_view

        if display_view.basis_button.isChecked():
            return basis[self._mode_spin_box.value() - 1]

        weight = series.opr_weight[index, :]

        if display_view.deviation_button.isChecked():
            weight = weight - series.opr_weight.mean(axis=0)

        return numpy.tensordot(weight, basis, axes=1)

    def _lock_display_range(self) -> None:
        """Fix the color axis over the whole scan so playback does not flicker.

        Sampling rather than sweeping every position keeps this cheap on a long scan, at
        the cost of clipping an unsampled outlier. The range suits the amplitude-mapping
        renderers the dialog opens on; the ribbon's Data Range controls override it.
        """
        if self._series is None or self._num_positions < 1:
            return

        try:
            samples = _subsample_indexes(self._num_positions, _NUM_DISPLAY_RANGE_SAMPLES)
            amplitude = max(
                float(numpy.absolute(self._get_field(int(index))).max()) for index in samples
            )
        except Exception as err:
            logger.exception(err)
            return

        if amplitude > 0.0:
            self._engine.set_display_value_range(0.0, amplitude)

    # ------------------------------------------------------------------ rendering

    def _update_image(self) -> None:
        if self._series is None or self._probes is None:
            self._image_controller.clear_array()
            return

        try:
            pixel_geometry = self._probes.get_pixel_geometry()
        except ValueError:
            logger.warning('Missing probe pixel geometry!')
            self._image_controller.clear_array()
            return

        try:
            field = self._get_field(self._dialog.position_slider.value())
        except IndexError:
            self._image_controller.clear_array()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Update Probe', err)
        else:
            self._image_controller.set_array(field, pixel_geometry)

    def _checked_modes(self) -> list[int]:
        widget = self._dialog.curves_view.mode_list_widget
        return [
            row
            for row in range(widget.count())
            if (item := widget.item(row)) is not None and item.checkState() == Qt.CheckState.Checked
        ]

    def _redraw_series(self) -> None:
        """Rebuild the series plot from scratch.

        The figure is rebuilt rather than cleared because ``twinx`` adds an axis on every
        call, which would otherwise accumulate. Only the position marker is touched per
        frame -- see :meth:`_update_markers`.
        """
        view = self._dialog.series_plot_view
        view.figure.clear()
        axes = view.axes = view.figure.add_subplot(111)
        self._position_marker = None
        self._series_marker.set_artist(None, None)

        series = self._series

        if series is None or self._num_positions < 1:
            view.figure_canvas.draw_idle()
            return

        position = numpy.arange(self._num_positions)
        artists = []

        (artist,) = axes.plot(
            position,
            series.primary_mode_photon_count,
            '-',
            linewidth=1.5,
            color='C0',
            label='Primary Mode',
        )
        artists.append(artist)
        (artist,) = axes.plot(
            position,
            series.total_photon_count,
            '-',
            linewidth=1.0,
            color='C1',
            label='Whole Probe',
        )
        artists.append(artist)
        axes.set_xlabel('Probe Position')
        axes.set_ylabel('Photon Count', color='C0')
        axes.tick_params(axis='y', labelcolor='C0')
        axes.grid(True)

        checked = self._checked_modes()

        if checked:
            weight_axes = axes.twinx()
            normalize = self._dialog.curves_view.normalize_check_box.isChecked()

            for mode in checked:
                weight = series.opr_weight[:, mode]
                value = weight - weight.mean()

                if normalize:
                    deviation = value.std()

                    if deviation > 0.0:
                        value = value / deviation

                color = f'C{2 + mode % 8}'
                (artist,) = weight_axes.plot(
                    position, value, '-', linewidth=1.0, color=color, label=f'OPR Mode {mode + 1}'
                )
                artists.append(artist)

            label = 'Weight (normalized)' if normalize else 'Weight − mean'
            weight_axes.set_ylabel(label)

        measured = self._measured_photon_count

        if measured is not None and measured.size == self._num_positions:
            measured_axes = axes.twinx()
            # Detector counts and probe-array power share no scale, so this needs its own
            # axis. It goes outboard on the left rather than stacking behind the weight
            # axis on the right, where the two y-labels would overlap.
            measured_axes.yaxis.set_label_position('left')
            measured_axes.yaxis.tick_left()
            measured_axes.spines['left'].set_position(('outward', 52))
            (artist,) = measured_axes.plot(
                position, measured, '.', markersize=2, color='0.4', label='Measured'
            )
            artists.append(artist)
            measured_axes.set_ylabel('Measured Counts', color='0.4')
            measured_axes.tick_params(axis='y', labelcolor='0.4')

        self._position_marker = axes.axvline(
            self._dialog.position_slider.value(), color='tab:red', linewidth=1
        )
        # Two columns and a translucent frame: the plot pane is short, so a tall opaque
        # legend would cover the curves it is labelling.
        axes.legend(
            artists,
            [str(artist.get_label()) for artist in artists],
            loc='best',
            fontsize='x-small',
            ncol=2,
            framealpha=0.6,
        )
        self._series_marker.set_artist(axes, self._position_marker)
        view.figure.tight_layout()
        view.figure_canvas.draw_idle()

    def _get_color_by(self) -> tuple[_ColorBy, int]:
        data = self._dialog.curves_view.color_by_combo_box.currentData()
        return data if data is not None else (_ColorBy.DEVIATION, 0)

    def _get_grid_values(self, num_points: int) -> tuple[RealArrayType, str]:
        series = self._series

        if series is None:
            return numpy.zeros(num_points), ''

        color_by, mode = self._get_color_by()

        match color_by:
            case _ColorBy.PRIMARY:
                return series.primary_mode_photon_count[:num_points], 'Primary Mode Counts'
            case _ColorBy.TOTAL:
                return series.total_photon_count[:num_points], 'Whole Probe Counts'
            case _ColorBy.MEASURED:
                measured = self._measured_photon_count
                values = numpy.zeros(num_points) if measured is None else measured[:num_points]
                return values, 'Measured Counts'
            case _ColorBy.WEIGHT:
                weight = series.opr_weight[:num_points, mode]
                return weight - weight.mean(), f'OPR Mode {mode + 1} Weight − mean'
            case _:
                return series.deviation_photon_count[:num_points], 'Deviation Power'

    def _redraw_scan_grid(self) -> None:
        view = self._dialog.scan_grid_plot_view
        view.figure.clear()
        axes = view.axes = view.figure.add_subplot(111)
        self._grid_highlight = None
        self._grid_marker.set_artist(None, None)
        self._grid_x_m = numpy.empty(0)
        self._grid_y_m = numpy.empty(0)

        if self._series is None or self._num_positions < 1:
            view.figure_canvas.draw_idle()
            return

        try:
            positions = self._analyzer.get_probe_positions(self._product_index)
        except Exception as err:
            logger.exception(err)
            view.figure_canvas.draw_idle()
            return

        # Weight rows pair with position rows, so a disagreement in length can only be
        # resolved by plotting the rows both sides have.
        num_points = min(self._num_positions, len(positions))

        if num_points < 1:
            view.figure_canvas.draw_idle()
            return

        self._grid_x_m = positions.get_coordinates_x_m()[:num_points]
        self._grid_y_m = positions.get_coordinates_y_m()[:num_points]
        x_m = self._grid_x_m
        y_m = self._grid_y_m
        values, label = self._get_grid_values(num_points)

        scatter = axes.scatter(x_m, y_m, c=values, s=12, cmap='viridis')
        (self._grid_highlight,) = axes.plot(
            [x_m[0]], [y_m[0]], 'o', markersize=11, markerfacecolor='none', color='tab:red'
        )
        view.figure.colorbar(scatter, ax=axes, label=label)
        axes.invert_yaxis()
        axes.axis('equal')
        axes.grid(True)
        axes.set_xlabel('X [m]')
        axes.set_ylabel('Y [m]')
        self._grid_marker.set_artist(axes, self._grid_highlight)
        view.figure.tight_layout()
        self._update_markers(self._dialog.position_slider.value())
        view.figure_canvas.draw_idle()

    def _update_markers(self, index: int) -> None:
        """Move the current-position markers without redrawing either plot.

        Rebuilding the curves and the scatter every frame would not keep up with playback
        over a long scan, so the static content is drawn once and only these two artists
        move.
        """
        if self._position_marker is not None:
            self._position_marker.set_xdata([index, index])
            self._series_marker.update()

        if self._grid_highlight is not None and index < self._grid_x_m.size:
            self._grid_highlight.set_data([self._grid_x_m[index]], [self._grid_y_m[index]])
            self._grid_marker.update()

    def _update_current_position(self, index: int) -> None:
        self._update_image()
        self._update_markers(index)

        if self._num_positions > 0:
            self._dialog.position_label.setText(f'Position {index + 1} / {self._num_positions}')
        else:
            self._dialog.position_label.setText('No Positions')

    # ------------------------------------------------------------------ playback

    def _get_frame_interval_ms(self) -> int:
        return max(1, round(1000.0 / self._frame_rate_spin_box.value()))

    def _stop_playback(self) -> None:
        """Idempotent: safe to call from the toggle handler's own re-entry."""
        self._playback_timer.stop()
        self._play_button.setChecked(False)

    def _handle_play_toggled(self) -> None:
        # Read the button rather than the signal's argument, and never call setChecked
        # here: a nested emission would let the outer frame's stale value win.
        is_playing = self._play_button.isChecked()

        if is_playing:
            self._playback_timer.start(self._get_frame_interval_ms())
        else:
            self._playback_timer.stop()

        if self._pause_icon is not None and self._play_icon is not None:
            self._play_button.setIcon(self._pause_icon if is_playing else self._play_icon)

        self._play_button.setText('Pause' if is_playing else 'Play')

    def _advance_frame(self) -> None:
        if not self._play_button.isChecked() or self._num_positions < 2:
            return

        slider = self._dialog.position_slider
        # Wrap on the position count, not the slider maximum: a degenerate range leaves
        # the maximum above the last valid index.
        next_value = 0 if slider.value() >= self._num_positions - 1 else slider.value() + 1
        slider.setValue(next_value)
        # Re-arm here rather than from the render slot. setValue emits nothing when the
        # value does not move, which is exactly what happens on the wrap frame, so a
        # re-arm inside the slot would stall playback after one lap.
        self._playback_timer.start(self._get_frame_interval_ms())

    def _handle_slider_pressed(self) -> None:
        self._was_playing_before_drag = self._play_button.isChecked()

        if self._was_playing_before_drag:
            self._stop_playback()

    def _handle_slider_released(self) -> None:
        if self._was_playing_before_drag and self._num_positions > 1:
            self._play_button.setChecked(True)

        self._was_playing_before_drag = False

    def _handle_dialog_finished(self, result: int) -> None:
        self._stop_playback()

    # ------------------------------------------------------------------ syncing

    def _handle_display_toggled(self) -> None:
        self._mode_spin_box.setEnabled(
            self._dialog.display_view.basis_button.isChecked() and self._mode_spin_box.maximum() > 1
        )
        self._lock_display_range()
        self._update_image()

    def _sync_statistics_view(self) -> None:
        view = self._dialog.statistics_view
        series = self._series

        if series is None:
            view.set_num_modes(0)
            view.clear_modes()
            return

        view.set_num_modes(series.num_coherent_modes)

        for mode, statistics in enumerate(series.mode_statistics):
            view.set_mode(
                mode,
                f'{statistics.mean_weight:.4g}',
                f'{statistics.weight_deviation:.4g}',
                f'{100.0 * statistics.variance_fraction:.1f}',
            )

        view.relative_variation_label.setText(f'{100.0 * series.relative_variation:.3g}')
        view.effective_mode_count_label.setText(f'{series.effective_mode_count}')

    def _sync_mode_controls(self) -> None:
        series = self._series
        num_modes = 1 if series is None else series.num_coherent_modes

        self._mode_spin_box.blockSignals(True)
        self._mode_spin_box.setRange(1, max(1, num_modes))
        self._mode_spin_box.blockSignals(False)
        self._mode_spin_box.setEnabled(
            self._dialog.display_view.basis_button.isChecked() and num_modes > 1
        )

        widget = self._dialog.curves_view.mode_list_widget
        widget.blockSignals(True)
        widget.clear()

        if series is not None:
            for mode in range(num_modes):
                item = QListWidgetItem(f'OPR Mode {mode + 1}')
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                # The first two modes up front: mode 1 carries the overall level and
                # mode 2 is where per-position structure first shows.
                item.setCheckState(Qt.CheckState.Checked if mode < 2 else Qt.CheckState.Unchecked)
                widget.addItem(item)

        widget.blockSignals(False)

        combo_box = self._dialog.curves_view.color_by_combo_box
        combo_box.blockSignals(True)
        combo_box.clear()

        if series is not None:
            combo_box.addItem('Deviation Power', (_ColorBy.DEVIATION, 0))
            combo_box.addItem('Primary Mode Counts', (_ColorBy.PRIMARY, 0))
            combo_box.addItem('Whole Probe Counts', (_ColorBy.TOTAL, 0))

            if self._measured_photon_count is not None:
                combo_box.addItem('Measured Counts', (_ColorBy.MEASURED, 0))

            for mode in range(num_modes):
                combo_box.addItem(f'OPR Mode {mode + 1} Weight', (_ColorBy.WEIGHT, mode))

        combo_box.blockSignals(False)

    def _sync_model_to_view(self) -> None:
        slider = self._dialog.position_slider
        slider.blockSignals(True)

        if self._num_positions > 1:
            slider.setEnabled(True)
            slider.setRange(0, self._num_positions - 1)
        else:
            slider.setEnabled(False)
            slider.setRange(0, max(0, self._num_positions - 1))

        slider.setValue(0)
        slider.blockSignals(False)

        self._play_button.setEnabled(self._num_positions > 1)
        self._dialog.save_button.setEnabled(self._series is not None)

        self._sync_mode_controls()
        self._sync_statistics_view()
        self._lock_display_range()
        self._redraw_series()
        self._redraw_scan_grid()
        self._update_current_position(0)

    # ------------------------------------------------------------------ saving

    def _save_series(self) -> None:
        series = self._series

        if series is None:
            logger.warning('No OPR mode series to save!')
            return

        title = 'Save OPR Mode Series'
        file_path, _name_filter = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=[_SAVE_FILE_FILTER],
            selected_name_filter=_SAVE_FILE_FILTER,
        )

        if file_path:
            try:
                series.save_npz(file_path)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)
