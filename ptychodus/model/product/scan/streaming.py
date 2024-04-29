from __future__ import annotations
from collections.abc import Sequence
import logging

from pvaccess import Channel

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder

logger = logging.getLogger(__name__)

# TODO class PositionStream:
# TODO
# TODO     def __init__(self) -> None:
# TODO         self._indexes: list[int] = list()
# TODO         self._positions: list[float] = list()
# TODO         self._minimum = +numpy.inf
# TODO         self._maximum = -numpy.inf
# TODO         self._uniqueness = 0
# TODO
# TODO     @property
# TODO     def minimum(self) -> float:
# TODO         return self._minimum
# TODO
# TODO     @property
# TODO     def maximum(self) -> float:
# TODO         return self._maximum
# TODO
# TODO     def append(self, index: int, position: float) -> None:
# TODO         lastIndex = self._indexes[-1]
# TODO
# TODO         if index > lastIndex:
# TODO             self._uniqueness = 0
# TODO             self._indexes.append(index)
# TODO             self._positions.append(position)
# TODO         elif index == lastIndex:  # online mean
# TODO             self._uniqueness += 1
# TODO             self._positions[-1] += (position - self._positions[-1]) / self._uniqueness
# TODO         else:  # index < lastIndex
# TODO             logger.warning(f'Discarding non-monotonic {index=}!')
# TODO
# TODO         if position < self._minimum:
# TODO             self._minimum = position
# TODO
# TODO         if position > self._maximum:
# TODO             self._maximum = position

# FIXME pvaccess.Channel.monitor
# FIXME PvObjectQueue


class StreamingScanBuilder(ScanBuilder):  # FIXME pvaccess

    def __init__(self, pointSeq: Sequence[ScanPoint]) -> None:
        super().__init__('Streaming')
        self._pointList = list(pointSeq)

    def append(self, point: ScanPoint) -> None:
        self._pointList.append(point)

    def extend(self, pointSeq: Sequence[ScanPoint]) -> None:
        self._pointList.extend(pointSeq)

    def build(self) -> Scan:
        return Scan(self._pointList)


# FIXME     def echo(self, value: int = 125) -> None:
# FIXME         print(f'{value=}')
# FIXME
# FIXME     def foo(self) -> None:
# FIXME         channelName = 'foo'
# FIXME         ch = Channel(channelName)
# FIXME         connected = ch.isConnected()
# FIXME         isActive = ch.isMonitorActive()
# FIXME         ch.subscribe('echo', self.echo)
# FIXME         ch.setMonitorMaxQueueLength(3)
# FIXME         ch.startMonitor()
# FIXME         ch.stopMonitor()
# FIXME         ch.unsubscribe('echo')
# FIXME
# FIXME         counterDict = ch.getMonitorCounters()
