import numpy

from ...api.parametric import Parameter
from ...api.probe import Probe
from .builder import FromMemoryProbeBuilder
from .item import ProbeRepositoryItem
from .multimodal import MultimodalProbeBuilder


class ProbeRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def create(self, name: Parameter[str], probe: Probe) -> ProbeRepositoryItem:
        builder = FromMemoryProbeBuilder(probe)
        multimodalBuilder = MultimodalProbeBuilder(self._rng)
        return ProbeRepositoryItem(name, builder, multimodalBuilder)
