from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader

logger = logging.getLogger(__name__)


class CXIProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> Probe:
        array = numpy.zeros((0, 0, 0), dtype=complex)

        with h5py.File(filePath, "r") as h5File:
            try:
                array = h5File["/entry_1/instrument_1/source_1/illumination"][()]
            except KeyError:
                logger.warning("Unable to load probe.")

        return Probe(array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.probeFileReaders.registerPlugin(
        CXIProbeFileReader(),
        simpleName="CXI",
        displayName="Coherent X-ray Imaging Files (*.cxi)",
    )
