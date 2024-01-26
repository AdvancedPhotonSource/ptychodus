import numpy

from ...api.plugins import PluginChooser
from ...api.probe import ProbeFileReader, ProbeFileWriter
from .builderFactory import ProbeBuilderFactory


class ProbeCore:

    def __init__(self, rng: numpy.random.Generator,
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        self._builderFactory = ProbeBuilderFactory(fileReaderChooser, fileWriterChooser)
