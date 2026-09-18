from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import Any
import logging
import math
import time

import numpy

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt
from PyQt5.QtWidgets import QApplication, QHeaderView, QWidget

from ptychodus.api.constants import LengthUnit, format_length
from ptychodus.api.probe import (
    FocalPlane,
    ProbeFocusCurves,
    ProbeFocusMetric,
    ProbeFocusSeries,
    compute_probe_focus_curves,
)
from ptychodus.api.propagate import PropagatedProbe
from ptychodus.api.typing import RealArrayType

from ...view.probe import ProbeFocusDialog
from ...view.widgets import ExceptionDialog

logger = logging.getLogger(__name__)

_PLACEHOLDER = '—'

# Metrics checked when the dialog first opens: one width, one intensity concentration,
# and one wavefront measure, so the default view already shows whether the three
# families agree about where the focus is.
_DEFAULT_METRICS = frozenset(
    {
        ProbeFocusMetric.FWHM_MAJOR,
        ProbeFocusMetric.FWHM_MINOR,
        ProbeFocusMetric.PEAK_INTENSITY,
        ProbeFocusMetric.PHASE_DEVIATION,
    }
)


def _format_scalar(value: float) -> str:
    if math.isnan(value):
        return _PLACEHOLDER

    if math.isinf(value):
        return '∞'

    return f'{value:.4g}'


def _display_suffix(metric: ProbeFocusMetric) -> str:
    """Unit shown after a metric's name, derived from the unit it is stored in."""
    match metric.si_unit:
        case 'm':
            return f' [{LengthUnit.NANOMETER.label}]'
        case '':
            return ''
        case unit:
            return f' [{unit}]'


def _display_scale(metric: ProbeFocusMetric) -> float:
    """Divisor taking a metric's stored SI samples into the unit :func:`_display_suffix`
    names.

    A divisor rather than a call to `LengthUnit.convert`, which is typed for scalars;
    dividing works the same for a single focal-plane value and for a whole curve.
    """
    if metric.si_unit == 'm':
        return LengthUnit.NANOMETER.meters_per_unit

    return 1.0


def compute_focus_curves(propagated_probe: PropagatedProbe, mode: int) -> ProbeFocusCurves:
    """Sweep the focus metrics, logging how long the sweep took.

    Timing is worth recording because the sweep is the one expensive thing this dialog
    does, and it grows with the step count. Unlike `compute_xy_metrics` in the sibling
    module, which runs on every slider move and so must swallow, this runs on an
    explicit button press -- a failure there should reach the user, so it propagates.
    """
    logger.info('Computing probe focus curves...')
    tic = time.perf_counter()
    curves = compute_probe_focus_curves(propagated_probe, mode=mode)
    toc = time.perf_counter()
    logger.info(f'Computed probe focus curves in {toc - tic:.4f} seconds.')
    return curves


class ProbeFocusTableModel(QAbstractTableModel):
    """Checkable list of focus metrics, each row carrying the focal plane it implies.

    Check state chooses what the plot draws; the view's current row chooses what the
    focus marker and the Go To Focus action operate on. Both are transient view state,
    so neither is persisted to settings.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._series: Sequence[ProbeFocusSeries] = []
        self._planes: dict[ProbeFocusMetric, FocalPlane] = {}
        self._checked: set[ProbeFocusMetric] = set(_DEFAULT_METRICS)
        self._header = ['Metric', 'Value', 'Focus']

    def set_curves(self, curves: ProbeFocusCurves | None) -> None:
        self.beginResetModel()

        if curves is None:
            self._series = []
            self._planes = {}
        else:
            self._series = curves.series
            self._planes = {s.metric: curves.get_focal_plane(s.metric) for s in curves.series}

        self.endResetModel()

    def get_metric(self, row: int) -> ProbeFocusMetric | None:
        # An empty selection reports row -1, which would otherwise index from the end
        # and silently hand back the last metric.
        if row < 0:
            return None

        try:
            return self._series[row].metric
        except IndexError:
            return None

    def get_focal_plane(self, metric: ProbeFocusMetric) -> FocalPlane | None:
        return self._planes.get(metric)

    def get_checked_series(self) -> Sequence[ProbeFocusSeries]:
        return [s for s in self._series if s.metric in self._checked]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() == 0:
            value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        metric = self.get_metric(index.row())

        if metric is None:
            return None

        if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if metric in self._checked else Qt.CheckState.Unchecked

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        match index.column():
            case 0:
                return metric.label + _display_suffix(metric)
            case 1:
                plane = self._planes.get(metric)
                return (
                    _PLACEHOLDER
                    if plane is None
                    else _format_scalar(plane.value / _display_scale(metric))
                )
            case 2:
                plane = self._planes.get(metric)
                return _PLACEHOLDER if plane is None else format_length(plane.coordinate_m)

        return None

    def setData(  # noqa: N802
        self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole
    ) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.CheckStateRole:
            return False

        metric = self.get_metric(index.row())

        if metric is None:
            return False

        if value == Qt.CheckState.Checked:
            self._checked.add(metric)
        else:
            self._checked.discard(metric)

        self.dataChanged.emit(index, index)
        return True

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._series)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)


class ProbeFocusViewController:
    def __init__(self, go_to_step: Callable[[int], None], parent: QWidget) -> None:
        super().__init__()
        self._go_to_step = go_to_step
        self._propagated_probe: PropagatedProbe | None = None
        self._curves: ProbeFocusCurves | None = None
        self._mode = -1

        self._table_model = ProbeFocusTableModel()
        self._dialog = ProbeFocusDialog(parent)
        self._dialog.setWindowTitle('Probe Focus')
        self._dialog.metric_table_view.setModel(self._table_model)

        # Only meaningful once the model supplies the columns; the metric name takes
        # the slack so the two numeric columns stay as narrow as their contents.
        header = self._dialog.metric_table_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self._table_model.dataChanged.connect(self._redraw_plot)
        self._dialog.normalize_check_box.toggled.connect(self._redraw_plot)
        self._dialog.go_to_focus_button.clicked.connect(self._go_to_focus)

        selection_model = self._dialog.metric_table_view.selectionModel()

        if selection_model is not None:
            selection_model.currentRowChanged.connect(self._redraw_plot)

    def analyze(self, propagated_probe: PropagatedProbe, mode: int) -> None:
        """Show the focus curves for *propagated_probe*, sweeping only when needed.

        The cached propagation is compared by identity: `PropagatedProbe` is a frozen
        dataclass of numpy arrays, so `==` would compare elementwise and then raise on
        the ambiguous truth value.
        """
        is_cached = propagated_probe is self._propagated_probe and mode == self._mode

        if not is_cached:
            application = QApplication.instance()

            if application is not None:
                application.setOverrideCursor(Qt.CursorShape.WaitCursor)  # type: ignore

            try:
                curves = compute_focus_curves(propagated_probe, mode)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('Probe Focus', err)
                return
            finally:
                if application is not None:
                    application.restoreOverrideCursor()  # type: ignore

            self._propagated_probe = propagated_probe
            self._curves = curves
            self._mode = mode
            self._table_model.set_curves(curves)
            self._select_default_row()

        self._redraw_plot()
        self._dialog.open()

    def invalidate(self) -> None:
        """Drop the cached sweep so the next `analyze` recomputes."""
        self._propagated_probe = None
        self._curves = None
        self._mode = -1
        self._table_model.set_curves(None)

    def _select_default_row(self) -> None:
        for row in range(self._table_model.rowCount()):
            if self._table_model.get_metric(row) in _DEFAULT_METRICS:
                self._dialog.metric_table_view.selectRow(row)
                return

        self._dialog.metric_table_view.selectRow(0)

    def _get_selected_metric(self) -> ProbeFocusMetric | None:
        return self._table_model.get_metric(self._dialog.metric_table_view.currentIndex().row())

    def _go_to_focus(self) -> None:
        metric = self._get_selected_metric()

        if metric is None:
            logger.warning('No metric selected!')
            return

        plane = self._table_model.get_focal_plane(metric)

        if plane is None:
            logger.warning(f'No focal plane for {metric}!')
            return

        self._go_to_step(plane.step)

    def _update_focus_label(self, metric: ProbeFocusMetric | None) -> None:
        plane = None if metric is None else self._table_model.get_focal_plane(metric)

        if metric is None or plane is None:
            self._dialog.focus_label.setText(_PLACEHOLDER)
            self._dialog.go_to_focus_button.setEnabled(False)
            return

        # Flag an unrefined estimate: it is the extremal sample itself, so it is only
        # as precise as the step size, and that is worth knowing before acting on it.
        qualifier = '' if plane.is_refined else ' (nearest plane)'
        self._dialog.focus_label.setText(
            f'{metric.label} at {format_length(plane.coordinate_m)}{qualifier}'
        )
        self._dialog.go_to_focus_button.setEnabled(True)

    def _redraw_plot(self) -> None:
        axes = self._dialog.axes
        axes.clear()

        selected_metric = self._get_selected_metric()
        self._update_focus_label(selected_metric)

        curves = self._curves
        checked = self._table_model.get_checked_series()

        if curves is None or not checked:
            axes.set_xlabel('Propagation Distance')
            self._dialog.figure.tight_layout()
            self._dialog.figure_canvas.draw()
            return

        # The axis the sweep actually sampled, not a reconstruction of it.
        coordinate_m = curves.coordinate_m
        unit = LengthUnit.from_meters(float(numpy.absolute(coordinate_m).max()))
        coordinate = coordinate_m / unit.meters_per_unit
        normalize = self._dialog.normalize_check_box.isChecked()

        for series in checked:
            metric = series.metric
            value = series.value / _display_scale(metric)
            label = metric.label

            if normalize:
                value = _normalize(value)
            else:
                label += _display_suffix(metric)

            axes.plot(coordinate, value, '.-', linewidth=1.5, label=label)

            plane = self._table_model.get_focal_plane(metric)

            if plane is not None:
                is_selected = metric is selected_metric
                axes.axvline(
                    plane.coordinate_m / unit.meters_per_unit,
                    color='0.3' if is_selected else '0.8',
                    linewidth=1.5 if is_selected else 1.0,
                    linestyle='--',
                    zorder=0,
                )

        axes.set_xlabel(f'Propagation Distance [{unit.label}]')
        axes.set_ylabel('Normalized Metric' if normalize else 'Metric')
        axes.grid(True)
        axes.legend(loc='best', fontsize='small')

        self._dialog.figure.tight_layout()
        self._dialog.figure_canvas.draw()


def _normalize(value: RealArrayType) -> RealArrayType:
    """Rescale a curve to ``[0, 1]`` for comparison against curves in other units.

    A constant curve carries no shape to compare, so it is drawn down the middle rather
    than divided by a zero range.
    """
    lower = value.min()
    upper = value.max()
    span = upper - lower

    if span <= 0.0:
        return numpy.full_like(value, 0.5)

    return (value - lower) / span
