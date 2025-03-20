from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader, ProbeFileWriter


class CSVProbeFileReader(ProbeFileReader):
    def read(self, file_path: Path) -> Probe:
        arrayFlat = numpy.genfromtxt(file_path, delimiter=',', dtype='complex')
        numberOfModes, remainder = divmod(arrayFlat.shape[0], arrayFlat.shape[1])

        if remainder != 0:
            raise ValueError('Failed to determine probe modes!')

        if numberOfModes > 1:
            array = arrayFlat.reshape(numberOfModes, arrayFlat.shape[1], arrayFlat.shape[1])

        return Probe(array=array, pixel_geometry=None)


class CSVProbeFileWriter(ProbeFileWriter):
    def write(self, file_path: Path, probe: Probe) -> None:
        array = probe.get_array()
        arrayFlat = array.reshape(-1, array.shape[-1])
        numpy.savetxt(file_path, arrayFlat, delimiter=',')


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_file_readers.register_plugin(
        CSVProbeFileReader(),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
    registry.probe_file_writers.register_plugin(
        CSVProbeFileWriter(),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
