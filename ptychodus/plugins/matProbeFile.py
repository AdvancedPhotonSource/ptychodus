from pathlib import Path

import numpy
import scipy.io

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter


class MATProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> ProbeArrayType:
        matDict = scipy.io.loadmat(filePath)
        probes = matDict['probe']
        return numpy.transpose(probes, [x for x in reversed(range(probes.ndim))])


class MATProbeFileWriter(ProbeFileWriter):

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        probes = numpy.transpose(array, [x for x in reversed(range(array.ndim))])
        matDict = {'probe': probes}
        scipy.io.savemat(filePath, matDict)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.probeFileReaders.registerPlugin(
        MATProbeFileReader(),
        simpleName='MAT',
        displayName='MAT Files (*.mat)',
    )
    registry.probeFileWriters.registerPlugin(
        MATProbeFileWriter(),
        simpleName='MAT',
        displayName='MAT Files (*.mat)',
    )
