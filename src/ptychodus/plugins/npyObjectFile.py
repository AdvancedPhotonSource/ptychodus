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


def register_plugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.register_plugin(
        NPYObjectFileReader(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
    registry.objectFileWriters.register_plugin(
        NPYObjectFileWriter(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
