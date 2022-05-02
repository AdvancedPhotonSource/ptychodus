from pathlib import Path

import scipy.io

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeFileReader, ProbeArrayType


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


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(MATProbeFileReader())
