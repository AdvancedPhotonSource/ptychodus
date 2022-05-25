from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter


class NPYProbeFileReader(ProbeFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def read(self, filePath: Path) -> ProbeArrayType:
        return numpy.load(filePath)


class NPYProbeFileWriter(ProbeFileWriter):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        numpy.save(filePath, array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPYProbeFileReader())
    registry.registerPlugin(NPYProbeFileWriter())
