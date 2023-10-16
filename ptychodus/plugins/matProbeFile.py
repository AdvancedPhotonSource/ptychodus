from pathlib import Path

import numpy
import scipy.io

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter


class MATProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> ProbeArrayType:
        matDict = scipy.io.loadmat(filePath)
        probes = matDict['probe']

        if probes.ndim == 4:
            # probes[width, height, num_shared_modes, num_varying_modes]
            probes = probes[..., 0]

        if probes.ndim == 3:
            # probes[width, height, num_shared_modes]
            probes = probes.transpose(2, 0, 1)

        return probes


class MATProbeFileWriter(ProbeFileWriter):

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        matDict = {'probe': array.transpose(1, 2, 0)}
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
