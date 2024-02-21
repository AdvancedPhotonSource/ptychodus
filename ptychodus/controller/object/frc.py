from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...model.analysis import FourierRingCorrelator
from ...view.object import FourierRingCorrelationDialog
from .listModel import ObjectListModel

logger = logging.getLogger(__name__)


class FourierRingCorrelationViewController:

    def __init__(self, correlator: FourierRingCorrelator,
                 dialog: FourierRingCorrelationDialog) -> None:
        super().__init__()
        self._correlator = correlator
        self._dialog = dialog

    @classmethod
    def analyze(cls, correlator: FourierRingCorrelator, listModel: ObjectListModel,
                parent: QWidget) -> FourierRingCorrelationDialog:
        dialog = FourierRingCorrelationDialog.createInstance(parent)
        viewController = cls(correlator, dialog)

        dialog.parametersView.name1ComboBox.setModel(listModel)
        dialog.parametersView.name1ComboBox.textActivated.connect(viewController._redrawPlot)
        dialog.parametersView.name2ComboBox.setModel(listModel)
        dialog.parametersView.name2ComboBox.textActivated.connect(viewController._redrawPlot)

        return dialog

    def _redrawPlot(self) -> None:
        currentIndex1 = self._dialog.parametersView.name1ComboBox.currentIndex()
        currentIndex2 = self._dialog.parametersView.name2ComboBox.currentIndex()

        if currentIndex1 < 0 or currentIndex2 < 0:
            logger.warning('Invalid item index for FRC!')
            return

        frc = self._correlator.correlate(currentIndex1, currentIndex2)
        plot2D = frc.getPlot()
        axisX = plot2D.axisX
        axisY = plot2D.axisY

        ax = self._dialog.axes
        ax.clear()
        ax.set_xlabel(axisX.label)
        ax.set_ylabel(axisY.label)
        ax.grid(True)

        if len(axisX.series) == 1:
            sx = axisX.series[0]

            for sy in axisY.series:
                ax.plot(sx.values, sy.values, '.-', label=sy.label, linewidth=1.5)
        else:
            logger.warning('Failed to broadcast plot series!')

        if len(axisX.series) > 1:
            ax.legend(loc='upper right')

        self._dialog.figureCanvas.draw()
