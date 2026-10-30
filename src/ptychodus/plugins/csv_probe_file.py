from pathlib import Path

import numpy

from ptychodus.api.files import FileFormat
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble, ProbeFileReader, ProbeFileWriter


class CSVProbeFileReader(ProbeFileReader):
    def get_file_description(self) -> str:
        return FileFormat.CSV.description

    def get_file_extensions(self) -> set[str]:
        return FileFormat.CSV.extensions

    def read(self, file_path: Path) -> ProbeEnsemble:
        array_flat = numpy.genfromtxt(file_path, delimiter=',', dtype=complex)
        num_modes, remainder = divmod(array_flat.shape[0], array_flat.shape[1])

        if remainder != 0:
            raise ValueError('Failed to determine probe modes!')

        array = array_flat.reshape(num_modes, array_flat.shape[1], array_flat.shape[1])
        return ProbeEnsemble(array=array, opr_weights=None, pixel_geometry=None)


class CSVProbeFileWriter(ProbeFileWriter):
    def get_file_description(self) -> str:
        return FileFormat.CSV.description

    def get_file_extensions(self) -> set[str]:
        return FileFormat.CSV.extensions

    def write(self, file_path: Path, probes: ProbeEnsemble) -> None:
        array = probes.get_probe_no_opr().get_array()
        array_flat = array.reshape(-1, array.shape[-1])
        numpy.savetxt(file_path, array_flat, delimiter=',')


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_file_readers.register_plugin(
        CSVProbeFileReader(),
        simple_name='CSV',
    )
    registry.probe_file_writers.register_plugin(
        CSVProbeFileWriter(),
        simple_name='CSV',
    )
