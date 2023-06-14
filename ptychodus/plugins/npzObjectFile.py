from pathlib import Path

import numpy

from ptychodus.api.object import ObjectArrayType, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.state import StateDataRegistry


class NPZObjectFileReader(ObjectFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPZ'

    @property
    def fileFilter(self) -> str:
        return StateDataRegistry.FILE_FILTER

    def read(self, filePath: Path) -> ObjectArrayType:
        npz = numpy.load(filePath)
        return npz[StateDataRegistry.OBJECT_ARRAY]


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPZObjectFileReader())
