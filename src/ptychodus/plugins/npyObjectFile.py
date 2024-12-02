from pathlib import Path

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class NPYObjectFileReader(ObjectFileReader):
    def read(self, filePath: Path) -> Object:
        array = numpy.load(filePath)
        return Object(array=array, pixelGeometry=None, center=None)


class NPYObjectFileWriter(ObjectFileWriter):
    def write(self, filePath: Path, object_: Object) -> None:
        array = object_.getArray()
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
