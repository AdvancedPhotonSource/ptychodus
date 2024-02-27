from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...model.analysis import FourierRingCorrelator
from ...view.object import FourierRingCorrelationDialog
from .listModel import ObjectListModel

logger = logging.getLogger(__name__)


class FourierRingCorrelationViewController:

    def __init__(self, correlator: FourierRingCorrelator, listModel: ObjectListModel,
                 parent: QWidget | None) -> None:
        super().__init__()
        self._correlator = correlator
        self._dialog = FourierRingCorrelationDialog.createInstance(parent)
        self._dialog.name1ComboBox.setModel(listModel)
        self._dialog.name1ComboBox.textActivated.connect(self._redrawPlot)
        self._dialog.name2ComboBox.setModel(listModel)
        self._dialog.name2ComboBox.textActivated.connect(self._redrawPlot)

    def analyze(self, itemIndex1: int, itemIndex2: int) -> None:
        self._dialog.name1ComboBox.setCurrentIndex(itemIndex1)
        self._dialog.name2ComboBox.setCurrentIndex(itemIndex2)
        self._redrawPlot()
        self._dialog.open()

    def _redrawPlot(self) -> None:
        currentIndex1 = self._dialog.name1ComboBox.currentIndex()
        currentIndex2 = self._dialog.name2ComboBox.currentIndex()

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
