from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QProgressBar

from ..model.memory import MemoryPresenter


class MemoryController:

    def __init__(self, presenter: MemoryPresenter, progressBar: QProgressBar,
                 timer: QTimer) -> None:
        self._presenter = presenter
        self._progressBar = progressBar
        self._timer = timer

    @classmethod
    def createInstance(cls, presenter: MemoryPresenter,
                       progressBar: QProgressBar) -> MemoryController:
        timer = QTimer()
        controller = cls(presenter, progressBar, timer)
        controller._updateProgressBar()

        timer.timeout.connect(controller._updateProgressBar)
        timer.start(10 * 1000)  # TODO customize (in milliseconds)

        return controller

    def _updateProgressBar(self) -> None:
        stats = self._presenter.getStatistics()
        totalMemMB = int(stats.totalMemoryInBytes / 1e6)
        totalMem = f'Total Memory: {totalMemMB} MB'

        availMemMB = int(stats.availableMemoryInBytes / 1e6)
        availMem = f'Available Memory: {availMemMB} MB'

        self._progressBar.setRange(0, 100)
        self._progressBar.setValue(int(stats.memoryUsagePercent))
        self._progressBar.setToolTip('\n'.join((totalMem, availMem)))
