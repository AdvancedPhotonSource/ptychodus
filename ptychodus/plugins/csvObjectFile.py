from pathlib import Path

import numpy

from ptychodus.api.object import ObjectArrayType, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class CSVObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> ObjectArrayType:
        return numpy.genfromtxt(filePath, delimiter=',', dtype='complex')


class CSVObjectFileWriter(ObjectFileWriter):

    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        numpy.savetxt(filePath, array, delimiter=',')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.registerPlugin(
        CSVObjectFileReader(),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
    registry.objectFileWriters.registerPlugin(
        CSVObjectFileWriter(),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
