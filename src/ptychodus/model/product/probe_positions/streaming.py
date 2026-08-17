from __future__ import annotations
from collections.abc import Sequence

import numpy

# TODO from pvaccess import Channel, PvObjectQueue

from ptychodus.api.probe_positions import ProbePosition

from .builder import ProbePositionsBuilder
from .settings import ProbePositionsSettings


class StreamingScanBuilder(ProbePositionsBuilder):
    # TODO The "discard at end" trim chases a moving tail while the stream is
    # still growing, so each build drops a different set of trailing points.
    # Decide on the semantics (most likely: honor the head trim, ignore the tail
    # trim until the stream is marked complete) before wiring up the pvaccess
    # path below.

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        point_seq: Sequence[ProbePosition],
    ) -> None:
        super().__init__(rng, settings, 'streaming')
        self._settings = settings
        self._point_list = list(point_seq)

    def copy(self) -> StreamingScanBuilder:
        builder = StreamingScanBuilder(self._rng, self._settings, self._point_list)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def append(self, point: ProbePosition) -> None:
        self._point_list.append(point)

    def extend(self, point_seq: Sequence[ProbePosition]) -> None:
        self._point_list.extend(point_seq)

    def _build_raw(self) -> Sequence[ProbePosition]:
        # Snapshot the list so a concurrent append cannot tear the trim.
        return [*self._point_list]


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
