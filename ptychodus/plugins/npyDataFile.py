from pathlib import Path

import numpy

from ptychodus.api.data import DataArrayType, DataFileReader, DataFileWriter
from ptychodus.api.plugins import PluginRegistry


class NPYDataFileWriter(DataFileWriter):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def write(self, filePath: Path, array: DataArrayType) -> None:
        numpy.save(filePath, array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPYDataFileWriter())
