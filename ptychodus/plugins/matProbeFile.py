from pathlib import Path

import scipy.io

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader, ProbeFileWriter


class MATProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> Probe:
        matDict = scipy.io.loadmat(filePath)
        array = matDict['probe']

        if array.ndim == 4:
            # array[width, height, num_shared_modes, num_varying_modes]
            array = array[..., 0]

        if array.ndim == 3:
            # array[width, height, num_shared_modes]
            array = array.transpose(2, 0, 1)

        return Probe(array)


class MATProbeFileWriter(ProbeFileWriter):

    def write(self, filePath: Path, probe: Probe) -> None:
        array = probe.getArray()
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
