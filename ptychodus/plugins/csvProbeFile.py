from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter


class CSVProbeFileReader(ProbeFileReader):
    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> ProbeArrayType:
        probe = numpy.genfromtxt(filePath, delimiter=',', dtype='complex')
        # TODO add size checks
        numberOfProbeModes = probe.shape[0] // probe.shape[1]

        if numberOfProbeModes > 1:
            probe = probe.reshape(numberOfProbeModes, probe.shape[1], probe.shape[1])

        return probe


class CSVProbeFileWriter(ProbeFileWriter):
    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        numpy.savetxt(filePath, array, delimiter=',')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CSVProbeFileReader())
    registry.registerPlugin(CSVProbeFileWriter())
