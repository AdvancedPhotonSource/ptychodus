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


def register_plugins(registry: PluginRegistry) -> None:
    registry.probeFileReaders.register_plugin(
        NPYProbeFileReader(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
    registry.probeFileWriters.register_plugin(
        NPYProbeFileWriter(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
