from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QProgressBar, QSizePolicy

from ..model.memory import MemoryPresenter


class MemoryController:
    def __init__(self, presenter: MemoryPresenter, progress_bar: QProgressBar) -> None:
        self._presenter = presenter
        self._progress_bar = progress_bar
        self._progress_bar.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self._timer = QTimer()
        self._timer.timeout.connect(self._updateProgressBar)

        self._updateProgressBar()
        self._timer.start(10 * 1000)  # TODO customize (in milliseconds)

    def _updateProgressBar(self) -> None:
        stats = self._presenter.getStatistics()
        totalMemMB = int(stats.totalMemoryInBytes / 1e6)
        totalMem = f'Total Memory: {totalMemMB} MB'

        availMemMB = int(stats.availableMemoryInBytes / 1e6)
        availMem = f'Available Memory: {availMemMB} MB'

        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(int(stats.memoryUsagePercent))
        self._progress_bar.setToolTip('\n'.join((totalMem, availMem)))
