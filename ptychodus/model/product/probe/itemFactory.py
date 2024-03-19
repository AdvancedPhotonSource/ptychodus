import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import FromMemoryProbeBuilder
from .disk import DiskProbeBuilder
from .item import ProbeRepositoryItem
from .multimodal import MultimodalProbeBuilder


class ProbeRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def create(self,
               geometryProvider: ProbeGeometryProvider,
               probe: Probe | None = None) -> ProbeRepositoryItem:
        builder = DiskProbeBuilder() if probe is None \
                else FromMemoryProbeBuilder(probe)
        multimodalBuilder = MultimodalProbeBuilder(self._rng)
        return ProbeRepositoryItem(geometryProvider, builder, multimodalBuilder)
