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
        return numpy.genfromtxt(filePath, delimiter=',', dtype='complex')


class CSVProbeFileWriter(ProbeFileWriter):
    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        numpy.savetxt(array, delimiter=',')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CSVProbeFileReader())
    registry.registerPlugin(CSVProbeFileWriter())
