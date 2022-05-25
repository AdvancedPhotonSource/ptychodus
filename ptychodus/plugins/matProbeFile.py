from pathlib import Path

import scipy.io

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter


class MATProbeFileReader(ProbeFileReader):

    @property
    def simpleName(self) -> str:
        return 'MAT'

    @property
    def fileFilter(self) -> str:
        return 'MAT Files (*.mat)'

    def read(self, filePath: Path) -> ProbeArrayType:
        matDict = scipy.io.loadmat(filePath)
        return matDict['probe']


class MATProbeFileWriter(ProbeFileWriter):

    @property
    def simpleName(self) -> str:
        return 'MAT'

    @property
    def fileFilter(self) -> str:
        return 'MAT Files (*.mat)'

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        matDict = {'probe': array}
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(MATProbeFileReader())
    registry.registerPlugin(MATProbeFileWriter())
