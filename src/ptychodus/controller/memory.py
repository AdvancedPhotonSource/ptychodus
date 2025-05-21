from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFrame, QLCDNumber, QSizePolicy

from ..model.memory import MemoryPresenter


class MemoryController:
    def __init__(self, presenter: MemoryPresenter, widget: QLCDNumber) -> None:
        self._presenter = presenter
        self._widget = widget
        self._widget.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self._widget.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self._widget.setDigitCount(6)
        self._widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_widget)

        self._update_widget()
        self._timer.start(10 * 1000)  # TODO customize (in milliseconds)

    def _update_widget(self) -> None:
        stats = self._presenter.get_statistics()
        total_MB = int(stats.total_physical_memory_bytes / 1e6)  # noqa: N806
        total_str = f'Total Memory: {total_MB} MB'

        avail_MB = int(stats.available_memory_bytes / 1e6)  # noqa: N806
        avail_str = f'Available Memory: {avail_MB} MB'

        self._widget.display(avail_MB)
        self._widget.setToolTip('\n'.join((total_str, avail_str)))
