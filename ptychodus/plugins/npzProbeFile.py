from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader
from ptychodus.api.state import StateDataRegistry


class NPZProbeFileReader(ProbeFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPZ'

    @property
    def fileFilter(self) -> str:
        return StateDataRegistry.FILE_FILTER

    def read(self, filePath: Path) -> ProbeArrayType:
        npz = numpy.load(filePath)
        return npz[StateDataRegistry.PROBE_ARRAY]


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPZProbeFileReader())
