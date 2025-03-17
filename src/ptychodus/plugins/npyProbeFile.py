from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader, ProbeFileWriter


class NPYProbeFileReader(ProbeFileReader):
    def read(self, filePath: Path) -> Probe:
        array = numpy.load(filePath)
        return Probe(array=array, pixel_geometry=None)


class NPYProbeFileWriter(ProbeFileWriter):
    def write(self, filePath: Path, probe: Probe) -> None:
        array = probe.get_array()
        numpy.save(filePath, array)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_file_readers.register_plugin(
        NPYProbeFileReader(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
    registry.probe_file_writers.register_plugin(
        NPYProbeFileWriter(),
        simple_name='NPY',
        display_name='NumPy Binary Files (*.npy)',
    )
