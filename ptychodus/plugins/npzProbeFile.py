from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader


class NPZProbeFileReader(ProbeFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPZ'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def read(self, filePath: Path) -> ProbeArrayType:
        npz = numpy.load(filePath)
        return npz['probe']


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPZProbeFileReader())
