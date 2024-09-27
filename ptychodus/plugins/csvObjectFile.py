from pathlib import Path

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class CSVObjectFileReader(ObjectFileReader):
    def read(self, filePath: Path) -> Object:
        array = numpy.genfromtxt(filePath, delimiter=",", dtype="complex")
        return Object(array)


class CSVObjectFileWriter(ObjectFileWriter):
    def write(self, filePath: Path, object_: Object) -> None:
        array = object_.array
        numpy.savetxt(filePath, array, delimiter=",")


def registerPlugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.registerPlugin(
        CSVObjectFileReader(),
        simpleName="CSV",
        displayName="Comma-Separated Values Files (*.csv)",
    )
    registry.objectFileWriters.registerPlugin(
        CSVObjectFileWriter(),
        simpleName="CSV",
        displayName="Comma-Separated Values Files (*.csv)",
    )
