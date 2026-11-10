from __future__ import annotations
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import math

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QComboBox, QTableWidgetItem

from matplotlib.axes import Axes
import numpy

from ptychodus.api.constants import format_length
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.metrics import PixelwiseMetrics
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.typing import RealArrayType

from ..view.dashboard import PLACEHOLDER, MetricsTableView


def format_scalar(value: float | None, *, suffix: str = '', precision: int = 3) -> str:
    """Format a value for display: ``None`` and NaN as the placeholder, infinity as ∞."""
    if value is None or math.isnan(value):
        return PLACEHOLDER

    if math.isinf(value):
        return f'{"-" if value < 0 else ""}∞{suffix}'

    return f'{value:.{precision}g}{suffix}'


def set_table_rows(table: MetricsTableView, rows: Sequence[tuple[str, Sequence[str]]]) -> None:
    """Replace the table's contents with one labelled row per entry of *rows*."""
    table.setRowCount(len(rows))
    table.setVerticalHeaderLabels([label for label, _ in rows])

    for row, (_, values) in enumerate(rows):
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            table.setItem(row, column, item)

    table.fit_to_contents()


def format_resampling(
    object_resampled_from: PixelGeometry | None, probe_resampled_from: ProbeGeometry | None
) -> str:
    """Describe what a comparison resampled the test product from, or say it did not."""
    parts: list[str] = []

    if object_resampled_from is not None:
        parts.append(f'object from {format_length(object_resampled_from.width_m)} pixels')

    if probe_resampled_from is not None:
        parts.append(
            f'probe from {probe_resampled_from.width_px} × {probe_resampled_from.height_px}'
            f' px at {format_length(probe_resampled_from.pixel_width_m)}'
        )

    return '; '.join(parts) if parts else 'No'


def pixelwise_rows(
    amplitude: PixelwiseMetrics, phase: PixelwiseMetrics
) -> list[tuple[str, list[str]]]:
    """Table rows comparing pixelwise amplitude and phase agreement side by side."""
    return [
        (
            'RMSE',
            [
                format_scalar(amplitude.root_mean_square_error),
                format_scalar(phase.root_mean_square_error),
            ],
        ),
        (
            'MAE',
            [
                format_scalar(amplitude.mean_absolute_error),
                format_scalar(phase.mean_absolute_error),
            ],
        ),
        (
            'PSNR [dB]',
            [
                format_scalar(amplitude.peak_signal_to_noise_ratio_dB),
                format_scalar(phase.peak_signal_to_noise_ratio_dB),
            ],
        ),
        (
            'SSIM',
            [
                format_scalar(amplitude.structural_similarity),
                format_scalar(phase.structural_similarity),
            ],
        ),
        (
            'NMI',
            [
                format_scalar(amplitude.normalized_mutual_information),
                format_scalar(phase.normalized_mutual_information),
            ],
        ),
    ]


def plot_histogram(axes: Axes, values: RealArrayType, *, max_bins: int = 50) -> None:
    """Histogram *values* on *axes*, in one bin when they span no more than roundoff."""
    finite = values[numpy.isfinite(values)]

    if finite.size == 0:
        return

    span = float(numpy.ptp(finite))
    scale = float(numpy.max(numpy.absolute(finite)))
    bins = max_bins if span > 1.0e-9 * scale else 1
    axes.hist(finite, bins=bins)


def select_pair(reference_combo_box: QComboBox, test_combo_box: QComboBox, index: int) -> None:
    """Select *index* as the reference and the next item, or *index* again, as the test."""
    reference_combo_box.setCurrentIndex(index)
    test_index = index + 1 if index + 1 < test_combo_box.count() else index
    test_combo_box.setCurrentIndex(test_index)


@contextmanager
def wait_cursor() -> Iterator[None]:
    """Show the busy cursor for the duration of the block."""
    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    try:
        yield
    finally:
        QGuiApplication.restoreOverrideCursor()
