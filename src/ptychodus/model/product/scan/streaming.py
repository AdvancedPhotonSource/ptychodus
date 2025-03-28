from collections.abc import Sequence

# TODO from pvaccess import Channel, PvObjectQueue

from ptychodus.api.scan import PositionSequence, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class StreamingScanBuilder(ScanBuilder):
    def __init__(self, settings: ScanSettings, point_seq: Sequence[ScanPoint]) -> None:
        super().__init__(settings, 'Streaming')
        self._point_list = list(point_seq)

    def append(self, point: ScanPoint) -> None:
        self._point_list.append(point)

    def extend(self, point_seq: Sequence[ScanPoint]) -> None:
        self._point_list.extend(point_seq)

    def build(self) -> PositionSequence:
        return PositionSequence(self._point_list)


# TODO def echo(self, value: int = 125) -> None:
# TODO     print(f'{value=}')
# TODO
# TODO def foo(self) -> None:
# TODO     channelName = 'foo'
# TODO     ch = Channel(channelName)
# TODO     connected = ch.isConnected()
# TODO     isActive = ch.isMonitorActive()
# TODO     ch.setMonitorMaxQueueLength(3)
# TODO     ch.subscribe('echo', self.echo)
# TODO     ch.startMonitor()
# TODO     ch.stopMonitor()
# TODO     ch.unsubscribe('echo')
# TODO
# TODO     counterDict = ch.getMonitorCounters()
