from pathlib import Path

import scipy.io

from ptychodus.api.object import ObjectArrayType, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class MATObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> ObjectArrayType:
        matDict = scipy.io.loadmat(filePath)
        object_ = matDict['object']

        if object_.ndim == 3:
            # object_[width, height, num_layers]
            object_ = object_.transpose(2, 0, 1)

        return object_


class MATObjectFileWriter(ObjectFileWriter):

    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        matDict = {'object': array.transpose(1, 2, 0)}
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.objectFileReaders.registerPlugin(
        MATObjectFileReader(),
        simpleName='MAT',
        displayName='MAT Files (*.mat)',
    )
    registry.objectFileWriters.registerPlugin(
        MATObjectFileWriter(),
        simpleName='MAT',
        displayName='MAT Files (*.mat)',
    )
