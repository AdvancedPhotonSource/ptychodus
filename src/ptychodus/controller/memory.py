from typing import Final

from PyQt5.QtWidgets import QFrame, QLCDNumber, QSizePolicy

from ptychodus.api.constants import ByteUnit, format_bytes

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
        total_str = f'Total Memory: {format_bytes(stats.total_physical_memory_bytes)}'
        avail_str = f'Available Memory: {format_bytes(stats.available_memory_bytes)}'

        # The LCD has six digits and no room for a unit suffix, so it stays in MB.
        self._widget.display(int(ByteUnit.MB.convert(stats.available_memory_bytes)))
        self._widget.setToolTip('\n'.join((total_str, avail_str)))
