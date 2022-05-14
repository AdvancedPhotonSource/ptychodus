from pathlib import Path

import scipy.io

from ptychodus.api.object import ObjectArrayType, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginRegistry


class MATObjectFileReader(ObjectFileReader):
    @property
    def simpleName(self) -> str:
        return 'MAT'

    @property
    def fileFilter(self) -> str:
        return 'MAT Files (*.mat)'

    def read(self, filePath: Path) -> ObjectArrayType:
        matDict = scipy.io.loadmat(filePath)
        return matDict['object']


class MATObjectFileWriter(ObjectFileWriter):
    @property
    def simpleName(self) -> str:
        return 'MAT'

    @property
    def fileFilter(self) -> str:
        return 'MAT Files (*.mat)'

    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        matDict = {'object': array}
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(MATObjectFileReader())
    registry.registerPlugin(MATObjectFileWriter())
