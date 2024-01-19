import numpy

from ...api.plugins import PluginChooser
from ...api.scan import ScanFileReader, ScanFileWriter
from .builderFactory import ScanBuilderFactory


class ScanCore:

    def __init__(self, rng: numpy.random.Generator,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._builderFactory = ScanBuilderFactory(fileReaderChooser, fileWriterChooser)
