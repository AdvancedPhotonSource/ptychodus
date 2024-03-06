import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import FromMemoryProbeBuilder
from .disk import DiskProbeBuilder
from .item import ProbeRepositoryItem
from .multimodal import MultimodalProbeBuilder


class ProbeRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def createDefault(self, geometryProvider: ProbeGeometryProvider) -> ProbeRepositoryItem:
        builder = DiskProbeBuilder(geometryProvider)
        multimodalBuilder = MultimodalProbeBuilder(self._rng)
        return ProbeRepositoryItem(builder, multimodalBuilder)

    def create(self, probe: Probe) -> ProbeRepositoryItem:
        builder = FromMemoryProbeBuilder(probe)
        multimodalBuilder = MultimodalProbeBuilder(self._rng)
        return ProbeRepositoryItem(builder, multimodalBuilder)
