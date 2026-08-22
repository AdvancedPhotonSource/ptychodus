from __future__ import annotations
import logging
import math

import numpy

from ptychodus.api.constants import LengthUnit

from ...model.analysis import FourierRingCorrelator
from ...view.object import FourierRingCorrelationDialog
from .tree_model import ObjectTreeModel

logger = logging.getLogger(__name__)

_SNR_THRESHOLD: float = 2.0  # equivalent to the classical FRC = 0.5 criterion
_PLACEHOLDER: str = '—'


def _format_scalar(value: float, *, suffix: str = '') -> str:
    if math.isnan(value):
        return _PLACEHOLDER
    if math.isinf(value):
        return f'∞{suffix}'
    return f'{value:.3g}{suffix}'


class FourierRingCorrelationViewController:
    def __init__(self, correlator: FourierRingCorrelator, tree_model: ObjectTreeModel) -> None:
        super().__init__()
        self._correlator = correlator
        self._dialog = FourierRingCorrelationDialog()
        self._dialog.setWindowTitle('Fourier Ring Correlation')
        self._dialog.product1_combo_box.setModel(tree_model)
        self._dialog.product1_combo_box.textActivated.connect(self._redraw_plot)
        self._dialog.product2_combo_box.setModel(tree_model)
        self._dialog.product2_combo_box.textActivated.connect(self._redraw_plot)

    def analyze(self, item_index1: int, item_index2: int) -> None:
        self._dialog.product1_combo_box.setCurrentIndex(item_index1)
        self._dialog.product2_combo_box.setCurrentIndex(item_index2)
        self._redraw_plot()
        self._dialog.open()

    def _clear_metric_labels(self) -> None:
        self._dialog.auc_label.setText(_PLACEHOLDER)
        self._dialog.average_ssnr_label.setText(_PLACEHOLDER)
        self._dialog.resolution_label.setText(_PLACEHOLDER)

    def _redraw_plot(self) -> None:
        current_index1 = self._dialog.product1_combo_box.currentIndex()
        current_index2 = self._dialog.product2_combo_box.currentIndex()

        if current_index1 < 0 or current_index2 < 0:
            logger.warning('Invalid item index for FRC!')
            self._clear_metric_labels()
            return

        frc = self._correlator.correlate(current_index1, current_index2)

        freq_per_nm = 1.0e-9 * frc.spatial_frequency_per_m
        ssnr = frc.get_spectral_signal_to_noise_ratio()
        # Log y-axis cannot render zeros — drop non-positive bins so they appear
        # as gaps instead of -inf warnings from matplotlib.
        ssnr_for_log = numpy.where(ssnr > 0.0, ssnr, numpy.nan)
        threshold_curve = frc.get_bit_threshold_curve(0.5)

        frc_axes = self._dialog.frc_axes
        ssnr_axes = self._dialog.ssnr_axes

        frc_axes.clear()
        frc_axes.set_ylabel('Fourier Ring Correlation')
        frc_axes.grid(True)
        frc_axes.plot(freq_per_nm, frc.correlation, '.-', linewidth=1.5, label='FRC')
        frc_axes.plot(freq_per_nm, threshold_curve, '--', linewidth=1.0, label='½-bit threshold')
        frc_axes.legend(loc='best', fontsize='small')
        frc_axes.tick_params(labelbottom=False)

        ssnr_axes.clear()
        ssnr_axes.set_xlabel('Spatial Frequency [1/nm]')
        ssnr_axes.set_ylabel('Spectral SNR')
        ssnr_axes.set_yscale('log')
        ssnr_axes.grid(True, which='both')
        ssnr_axes.plot(freq_per_nm, ssnr_for_log, '.-', linewidth=1.5)

        self._dialog.auc_label.setText(_format_scalar(frc.get_area_under_curve()))
        self._dialog.average_ssnr_label.setText(
            _format_scalar(frc.get_average_signal_to_noise_ratio())
        )
        resolution_m = frc.get_resolution_m_at_signal_to_noise_threshold(_SNR_THRESHOLD)
        self._dialog.resolution_label.setText(
            _format_scalar(LengthUnit.NANOMETER.convert(resolution_m), suffix=' nm')
        )

        self._dialog.figure.tight_layout()
        self._dialog.figure_canvas.draw()
