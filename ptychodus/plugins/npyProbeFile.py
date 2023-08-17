from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter


class NPYProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> ProbeArrayType:
        return numpy.load(filePath)


class NPYProbeFileWriter(ProbeFileWriter):

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
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
