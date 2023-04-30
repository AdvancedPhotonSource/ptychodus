from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeArrayType, ProbeFileReader

logger = logging.getLogger(__name__)


class CXIProbeFileReader(ProbeFileReader):

    @property
    def simpleName(self) -> str:
        return 'CXI'

    @property
    def fileFilter(self) -> str:
        return 'Coherent X-ray Imaging Files (*.cxi)'

    def read(self, filePath: Path) -> ProbeArrayType:
        probe = numpy.zeros((0, 0, 0), dtype=complex)

        with h5py.File(filePath, 'r') as h5File:
            try:
                probe = h5File['/entry_1/instrument_1/source_1/illumination'][()]
            except KeyError:
                logger.debug('Unable to load probe.')

        return probe


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CXIProbeFileReader())
