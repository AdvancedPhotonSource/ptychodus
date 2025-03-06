from pathlib import Path

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class CSVObjectFileReader(ObjectFileReader):
    def read(self, filePath: Path) -> Object:
        array = numpy.genfromtxt(filePath, delimiter=',', dtype='complex')
        return Object(array=array, pixelGeometry=None, center=None)


class CSVObjectFileWriter(ObjectFileWriter):
    def write(self, filePath: Path, object_: Object) -> None:
        array = object_.getArray()
        numpy.savetxt(filePath, array, delimiter=',')


def register_plugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.register_plugin(
        CSVObjectFileReader(),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
    registry.objectFileWriters.register_plugin(
        CSVObjectFileWriter(),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
