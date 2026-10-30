from __future__ import annotations
from enum import Enum, IntEnum
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


class _Row(Enum):
    """Every row of ProbeMetricsTableModel: its order, its label and its source.

    The member value pairs the label with the metrics object that supplies the value,
    and the position comes from declaration order rather than being written out, so a
    row cannot be added to the ordering and forgotten in the labels. Nothing could pin
    that alignment: CLAUDE.md puts this file outside the tested surface.

    Naming the source also drops the requirement that every size row precede every
    entropy row, which a ``row < 7`` test had quietly imposed.
    """

    MAJOR_AXIS_TILT_DEG = ('Major Axis Tilt [deg]', True)
    MINOR_AXIS_TILT_DEG = ('Minor Axis Tilt [deg]', True)
    FWHM_MAJOR_AXIS_NM = ('FWHM Major Axis [nm]', True)
    FWHM_MINOR_AXIS_NM = ('FWHM Minor Axis [nm]', True)
    RMS_MAJOR_AXIS_NM = ('RMS Major Axis [nm]', True)
    RMS_MINOR_AXIS_NM = ('RMS Minor Axis [nm]', True)
    ENCIRCLED_ENERGY_DIAMETER_NM = ('Encircled Energy Diameter [nm]', True)
    REAL_SPACE_INTENSITY_ENTROPY = ('Real-Space Intensity Entropy [bits, norm]', False)
    SPECTRAL_ENTROPY = ('Spectral Entropy [bits, norm]', False)

    def __init__(self, label: str, from_size_metrics: bool) -> None:
        self.label = label
        self.from_size_metrics = from_size_metrics
        # Members already defined when this one is constructed, which is its position.
        self.index = len(self.__class__.__members__)

    @classmethod
    def from_index(cls, index: int) -> _Row:
        """The row at `index`.

        Raises:
            IndexError: for a row outside the table, which the model's own
                ``rowCount`` makes impossible and so would be a bug rather than input.
        """
        if index < 0:
            raise IndexError(f'Row index {index} is negative.')

        return list(cls)[index]


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

        row = _Row.from_index(index.row())

        if index.column() == _Col.PROPERTY:
            return row.label

        if row.from_size_metrics:
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
        return len(_Row)

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
