from pathlib import Path

import numpy

from ptychodus.api.object import ObjectArrayType, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry


class NPZObjectFileReader(ObjectFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPZ'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def read(self, filePath: Path) -> ObjectArrayType:
        npz = numpy.load(filePath)
        return npz['object']


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPZObjectFileReader())
