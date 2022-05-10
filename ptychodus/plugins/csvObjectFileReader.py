from pathlib import Path

import numpy

from ptychodus.api.object import ObjectFileReader, ObjectArrayType
from ptychodus.api.plugins import PluginRegistry


class CSVObjectFileReader(ObjectFileReader):
    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> ObjectArrayType:
        return numpy.genfromtxt(filePath, delimiter=',', dtype='complex')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CSVObjectFileReader())
