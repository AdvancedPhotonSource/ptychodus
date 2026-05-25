from __future__ import annotations
import logging

from ...model.analysis import FourierRingCorrelator
from ...view.object import FourierRingCorrelationDialog
from .tree_model import ObjectTreeModel

logger = logging.getLogger(__name__)


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

    def _redraw_plot(self) -> None:
        current_index1 = self._dialog.product1_combo_box.currentIndex()
        current_index2 = self._dialog.product2_combo_box.currentIndex()

        if current_index1 < 0 or current_index2 < 0:
            logger.warning('Invalid item index for FRC!')
            return

        frc = self._correlator.correlate(current_index1, current_index2)

        ax = self._dialog.axes
        ax.clear()
        ax.set_xlabel('Spatial Frequency [1/nm]')
        ax.set_ylabel('Fourier Ring Correlation')
        ax.grid(True)
        ax.plot(
            1.0e-9 * frc.spatial_frequency_per_m,
            frc.correlation,
            '.-',
            linewidth=1.5,
        )

        self._dialog.figure_canvas.draw()
