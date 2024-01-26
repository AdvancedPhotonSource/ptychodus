import numpy

from ...api.probe import Probe
from .builder import FromMemoryProbeBuilder
from .item import ProbeRepositoryItem
from .multimodal import MultimodalProbeBuilder


class ProbeRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def create(self, probe: Probe) -> ProbeRepositoryItem:
        builder = FromMemoryProbeBuilder(probe)
        multimodalBuilder = MultimodalProbeBuilder(self._rng)
        return ProbeRepositoryItem(builder, multimodalBuilder)
