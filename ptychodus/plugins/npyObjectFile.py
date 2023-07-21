from pathlib import Path

import numpy

from ptychodus.api.object import ObjectArrayType, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class NPYObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> ObjectArrayType:
        return numpy.load(filePath)


class NPYObjectFileWriter(ObjectFileWriter):

    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        numpy.save(filePath, array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.registerPlugin(
        NPYObjectFileReader(),
        simpleName='NPY',
        displayName='NumPy Binary Files (*.npy)',
    )
    registry.objectFileWriters.registerPlugin(
        NPYObjectFileWriter(),
        simpleName='NPY',
        displayName='NumPy Binary Files (*.npy)',
    )
