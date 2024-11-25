from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader, ProbeFileWriter


class NPYProbeFileReader(ProbeFileReader):
    def read(self, filePath: Path) -> Probe:
        array = numpy.load(filePath)
        return Probe(array=array, pixelGeometry=None)


class NPYProbeFileWriter(ProbeFileWriter):
    def write(self, filePath: Path, probe: Probe) -> None:
        array = probe.getArray()
        numpy.save(filePath, array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.probeFileReaders.registerPlugin(
        NPYProbeFileReader(),
        simpleName='NPY',
        displayName='NumPy Binary Files (*.npy)',
    )
    registry.probeFileWriters.registerPlugin(
        NPYProbeFileWriter(),
        simpleName='NPY',
        displayName='NumPy Binary Files (*.npy)',
    )
