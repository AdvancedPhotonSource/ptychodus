from pathlib import Path

import numpy

from ptychodus.api.object import ObjectFileReader, ObjectArrayType
from ptychodus.api.plugins import PluginRegistry


class NPYObjectFileReader(ObjectFileReader):
    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def read(self, filePath: Path) -> ObjectArrayType:
        return numpy.load(filePath)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPYObjectFileReader())
