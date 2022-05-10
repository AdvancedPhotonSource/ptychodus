from pathlib import Path

import scipy.io

from ptychodus.api.object import ObjectFileReader, ObjectArrayType
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


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(MATObjectFileReader())
