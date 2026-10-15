from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble, ProbeFileReader, ProbeFileWriter


class NPYProbeFileReader(ProbeFileReader):
    def read(self, file_path: Path) -> ProbeEnsemble:
        array = numpy.load(file_path)
        return ProbeEnsemble(array=array, opr_weights=None, pixel_geometry=None)


class NPYProbeFileWriter(ProbeFileWriter):
    def write(self, file_path: Path, probes: ProbeEnsemble) -> None:
        array = probes.get_probe_no_opr().get_array()
        numpy.save(file_path, array)


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
