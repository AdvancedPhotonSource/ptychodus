from __future__ import annotations
from enum import IntEnum
from typing import Any
import logging

import numpy

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt

from ptychodus.api.constants import LengthUnit
from ptychodus.api.typing import RealArrayType
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import (
    Probe,
    ProbeEntropyMetrics,
    ProbeSizeMetrics,
    estimate_probe_entropy,
    estimate_probe_size,
)

logger = logging.getLogger(__name__)


class _Row(IntEnum):
    """Row order of ProbeMetricsTableModel.

    An enum rather than module-level ints so that ``match`` arms are value patterns:
    a bare name would be a capture pattern that swallows every row.
    """

    MAJOR_AXIS_TILT_DEG = 0
    MINOR_AXIS_TILT_DEG = 1
    FWHM_MAJOR_AXIS_NM = 2
    FWHM_MINOR_AXIS_NM = 3
    RMS_MAJOR_AXIS_NM = 4
    RMS_MINOR_AXIS_NM = 5
    ENCIRCLED_ENERGY_DIAMETER_NM = 6
    REAL_SPACE_INTENSITY_ENTROPY = 7
    SPECTRAL_ENTROPY = 8


class _Col(IntEnum):
    """Column order of ProbeMetricsTableModel.

    An enum for the same reason as _Row: bare names in a ``match`` arm are capture
    patterns, not value patterns.
    """

    PROPERTY = 0
    VALUE = 1


class ProbeMetricsTableModel(QAbstractTableModel):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._metrics: ProbeSizeMetrics | None = None
        self._entropy_metrics: ProbeEntropyMetrics | None = None
        self._header = ['Property', 'Value']
        self._properties = [
            'Major Axis Tilt [deg]',
            'Minor Axis Tilt [deg]',
            'FWHM Major Axis [nm]',
            'FWHM Minor Axis [nm]',
            'RMS Major Axis [nm]',
            'RMS Minor Axis [nm]',
            'Encircled Energy Diameter [nm]',
            'Real-Space Intensity Entropy [bits, norm]',
            'Spectral Entropy [bits, norm]',
        ]

    def set_metrics(self, metrics: ProbeSizeMetrics | None) -> None:
        self.beginResetModel()
        self._metrics = metrics
        self.endResetModel()

    def set_entropy_metrics(self, metrics: ProbeEntropyMetrics | None) -> None:
        self.beginResetModel()
        self._entropy_metrics = metrics
        self.endResetModel()

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None

        if index.column() == _Col.PROPERTY:
            return self._properties[index.row()]

        row = index.row()

        if row < _Row.REAL_SPACE_INTENSITY_ENTROPY:
            if self._metrics is None:
                return 'N/A'

            match row:
                case _Row.MAJOR_AXIS_TILT_DEG:
                    return f'{numpy.rad2deg(self._metrics.major_axis_tilt_rad):.4g}'
                case _Row.MINOR_AXIS_TILT_DEG:
                    return f'{numpy.rad2deg(self._metrics.minor_axis_tilt_rad):.4g}'
                case _Row.FWHM_MAJOR_AXIS_NM:
                    return f'{LengthUnit.NANOMETER.convert(self._metrics.fwhm_major_axis_length_m):.4g}'
                case _Row.FWHM_MINOR_AXIS_NM:
                    return f'{LengthUnit.NANOMETER.convert(self._metrics.fwhm_minor_axis_length_m):.4g}'
                case _Row.RMS_MAJOR_AXIS_NM:
                    return (
                        f'{LengthUnit.NANOMETER.convert(self._metrics.rms_major_axis_length_m):.4g}'
                    )
                case _Row.RMS_MINOR_AXIS_NM:
                    return (
                        f'{LengthUnit.NANOMETER.convert(self._metrics.rms_minor_axis_length_m):.4g}'
                    )
                case _Row.ENCIRCLED_ENERGY_DIAMETER_NM:
                    return f'{LengthUnit.NANOMETER.convert(self._metrics.encircled_energy_diameter_m):.4g}'
        else:
            if self._entropy_metrics is None:
                return 'N/A'

            match row:
                case _Row.REAL_SPACE_INTENSITY_ENTROPY:
                    return f'{self._entropy_metrics.real_space_intensity_entropy:.4g}'
                case _Row.SPECTRAL_ENTROPY:
                    return f'{self._entropy_metrics.spectral_entropy:.4g}'

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)


def compute_xy_metrics(
    intensity: RealArrayType, pixel_geometry: PixelGeometry
) -> ProbeSizeMetrics | None:
    try:
        return estimate_probe_size(intensity, pixel_geometry)
    except Exception:
        logger.exception('Failed to estimate probe size from XY intensity!')
        return None


def compute_entropy_metrics(probe: Probe) -> ProbeEntropyMetrics | None:
    try:
        return estimate_probe_entropy(probe)
    except Exception:
        logger.exception('Failed to estimate probe entropy!')
        return None
