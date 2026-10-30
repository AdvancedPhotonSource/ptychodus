from pathlib import Path

import numpy

from ptychodus.api.files import FileFormat
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble, ProbeFileReader, ProbeFileWriter


class NPYProbeFileReader(ProbeFileReader):
    def get_file_description(self) -> str:
        return FileFormat.NPY.description

    def get_file_extensions(self) -> set[str]:
        return FileFormat.NPY.extensions

    def read(self, file_path: Path) -> ProbeEnsemble:
        array = numpy.load(file_path)
        return ProbeEnsemble(array=array, opr_weights=None, pixel_geometry=None)


class NPYProbeFileWriter(ProbeFileWriter):
    def get_file_description(self) -> str:
        return FileFormat.NPY.description

    def get_file_extensions(self) -> set[str]:
        return FileFormat.NPY.extensions

    def write(self, file_path: Path, probes: ProbeEnsemble) -> None:
        array = probes.get_probe_no_opr().get_array()
        numpy.save(file_path, array)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_file_readers.register_plugin(
        NPYProbeFileReader(),
        simple_name='NPY',
    )
    registry.probe_file_writers.register_plugin(
        NPYProbeFileWriter(),
        simple_name='NPY',
    )
