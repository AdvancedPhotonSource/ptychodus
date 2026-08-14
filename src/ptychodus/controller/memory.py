from typing import Final

from PyQt5.QtWidgets import QFrame, QLCDNumber, QSizePolicy

from ptychodus.api.constants import BYTES_PER_MEGABYTE

from ..model.memory import MemoryPresenter


class MemoryController:
    UPDATE_INTERVAL_S: Final[int] = 10

    def __init__(self, presenter: MemoryPresenter, widget: QLCDNumber) -> None:
        self._presenter = presenter
        self._widget = widget
        self._widget.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self._widget.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        self._widget.setDigitCount(6)
        self._widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)

    def run_tasks(self, one_second_counter: int) -> None:
        if one_second_counter % MemoryController.UPDATE_INTERVAL_S != 0:
            return

        stats = self._presenter.get_statistics()
        total_MB = int(stats.total_physical_memory_bytes / BYTES_PER_MEGABYTE)  # noqa: N806
        total_str = f'Total Memory: {total_MB} MB'

        avail_MB = int(stats.available_memory_bytes / BYTES_PER_MEGABYTE)  # noqa: N806
        avail_str = f'Available Memory: {avail_MB} MB'

        self._widget.display(avail_MB)
        self._widget.setToolTip('\n'.join((total_str, avail_str)))
