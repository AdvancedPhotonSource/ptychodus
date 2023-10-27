from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.state import StateDataRegistry


class NPZProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> Probe:
        npz = numpy.load(filePath)
        array = npz[StateDataRegistry.PROBE_ARRAY]
        return Probe(array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.probeFileReaders.registerPlugin(
        NPZProbeFileReader(),
        simpleName='NPZ',
        displayName=StateDataRegistry.FILE_FILTER,
    )
