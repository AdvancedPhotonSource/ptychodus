from pathlib import Path

import numpy

from ptychodus.api.files import FileFormat
from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class CSVObjectFileReader(ObjectFileReader):
    def get_file_description(self) -> str:
        return FileFormat.CSV.description

    def get_file_extensions(self) -> set[str]:
        return FileFormat.CSV.extensions

    def read(self, file_path: Path) -> Object:
        array = numpy.genfromtxt(file_path, delimiter=',', dtype=complex)
        return Object(array=array, pixel_geometry=None, center=None)


class CSVObjectFileWriter(ObjectFileWriter):
    def get_file_description(self) -> str:
        return FileFormat.CSV.description

    def get_file_extensions(self) -> set[str]:
        return FileFormat.CSV.extensions

    def write(self, file_path: Path, object_: Object) -> None:
        array = object_.get_array()
        numpy.savetxt(file_path, array, delimiter=',')


def register_plugins(registry: PluginRegistry) -> None:
    registry.object_file_readers.register_plugin(
        CSVObjectFileReader(),
        simple_name='CSV',
    )
    registry.object_file_writers.register_plugin(
        CSVObjectFileWriter(),
        simple_name='CSV',
    )
