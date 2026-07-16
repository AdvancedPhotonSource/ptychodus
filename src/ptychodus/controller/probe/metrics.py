from typing import Any
import logging

import numpy

from PyQt5.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt

from ptychodus.api.common import RealArrayType
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import (
    Probe,
    ProbeEntropyMetrics,
    ProbeSizeMetrics,
    estimate_probe_entropy,
    estimate_probe_size,
)

logger = logging.getLogger(__name__)


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

        if index.column() == 0:
            return self._properties[index.row()]

        row = index.row()

        if row < 7:
            if self._metrics is None:
                return 'N/A'

            match row:
                case 0:
                    return f'{numpy.rad2deg(self._metrics.major_axis_tilt_rad):.4g}'
                case 1:
                    return f'{numpy.rad2deg(self._metrics.minor_axis_tilt_rad):.4g}'
                case 2:
                    return f'{self._metrics.fwhm_major_axis_length_m * 1e9:.4g}'
                case 3:
                    return f'{self._metrics.fwhm_minor_axis_length_m * 1e9:.4g}'
                case 4:
                    return f'{self._metrics.rms_major_axis_length_m * 1e9:.4g}'
                case 5:
                    return f'{self._metrics.rms_minor_axis_length_m * 1e9:.4g}'
                case 6:
                    return f'{self._metrics.encircled_energy_diameter_m * 1e9:.4g}'
        else:
            if self._entropy_metrics is None:
                return 'N/A'

            match row:
                case 7:
                    return f'{self._entropy_metrics.real_space_intensity_entropy:.4g}'
                case 8:
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
        logger.exception('Failed to estimate probe size from XY projection!')
        return None


def compute_entropy_metrics(probe: Probe) -> ProbeEntropyMetrics | None:
    try:
        return estimate_probe_entropy(probe)
    except Exception:
        logger.exception('Failed to estimate probe entropy!')
        return None
