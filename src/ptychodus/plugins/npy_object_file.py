from pathlib import Path

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class NPYObjectFileReader(ObjectFileReader):
    def read(self, file_path: Path) -> Object:
        array = numpy.load(file_path)
        return Object(array=array, pixel_geometry=None, center=None)


class NPYObjectFileWriter(ObjectFileWriter):
    def write(self, file_path: Path, object_: Object) -> None:
        array = object_.get_array()
        numpy.save(file_path, array)


def register_plugins(registry: PluginRegistry) -> None:
    registry.object_file_readers.register_plugin(
        NPYObjectFileReader(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
    registry.object_file_writers.register_plugin(
        NPYObjectFileWriter(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
