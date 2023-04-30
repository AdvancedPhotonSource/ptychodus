from pathlib import Path

import numpy
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
        probes = numpy.moveaxis(matDict['probe'], -1, 0)
        return probes


class MATProbeFileWriter(ProbeFileWriter):

    @property
    def simpleName(self) -> str:
        return 'MAT'

    @property
    def fileFilter(self) -> str:
        return 'MAT Files (*.mat)'

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        probes = numpy.moveaxis(array, 0, -1)
        matDict = {'probe': probes}
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(MATProbeFileReader())
    registry.registerPlugin(MATProbeFileWriter())
