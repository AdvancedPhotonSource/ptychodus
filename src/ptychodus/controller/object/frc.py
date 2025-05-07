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
        plot2d = frc.get_plot()
        axis_x = plot2d.axis_x
        axis_y = plot2d.axis_y

        ax = self._dialog.axes
        ax.clear()
        ax.set_xlabel(axis_x.label)
        ax.set_ylabel(axis_y.label)
        ax.grid(True)

        if len(axis_x.series) == 1:
            sx = axis_x.series[0]

            for sy in axis_y.series:
                ax.plot(sx.values, sy.values, '.-', label=sy.label, linewidth=1.5)
        else:
            logger.warning('Failed to broadcast plot series!')

        if len(axis_x.series) > 1:
            ax.legend(loc='upper right')

        self._dialog.figure_canvas.draw()
