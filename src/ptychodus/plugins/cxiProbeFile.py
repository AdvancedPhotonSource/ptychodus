from pathlib import Path
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.propagator import WavefieldArrayType

logger = logging.getLogger(__name__)


class CXIProbeFileReader(ProbeFileReader):
    def read(self, filePath: Path) -> Probe:
        array: WavefieldArrayType | None = None

        with h5py.File(filePath, 'r') as h5File:
            try:
                array = h5File['/entry_1/instrument_1/source_1/illumination'][()]
            except KeyError:
                logger.warning('Unable to load probe.')

        return Probe(array=array, pixelGeometry=None)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.probeFileReaders.registerPlugin(
        CXIProbeFileReader(),
        simpleName='CXI',
        displayName='Coherent X-ray Imaging Files (*.cxi)',
    )
