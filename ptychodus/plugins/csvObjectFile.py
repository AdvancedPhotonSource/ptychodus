from pathlib import Path

import numpy

from ptychodus.api.object import ObjectArrayType, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class CSVObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> ObjectArrayType:
        arrayFlat = numpy.genfromtxt(filePath, delimiter=',', dtype='complex')
        numberOfSlices, remainder = divmod(arrayFlat.shape[0], arrayFlat.shape[1])

        if remainder != 0:
            raise ValueError('Failed to determine object slices!')

        if numberOfSlices > 1:
            array = arrayFlat.reshape(numberOfSlices, arrayFlat.shape[1], arrayFlat.shape[1])

        return array


class CSVObjectFileWriter(ObjectFileWriter):

    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        arrayFlat = array.reshape(-1, array.shape[-1])
        numpy.savetxt(filePath, arrayFlat, delimiter=',')


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
