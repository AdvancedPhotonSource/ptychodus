from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeFileReader, ProbeArrayType


class CSVProbeFileReader(ProbeFileReader):
    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> ProbeArrayType:
        return numpy.genfromtxt(filePath, delimiter=',', dtype='complex')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CSVProbeFileReader())
