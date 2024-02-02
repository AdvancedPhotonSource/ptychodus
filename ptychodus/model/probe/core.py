import numpy

from ...api.plugins import PluginChooser
from ...api.probe import ProbeFileReader, ProbeFileWriter
from .builderFactory import ProbeBuilderFactory
from .itemFactory import ProbeRepositoryItemFactory


class ProbeCore:

    def __init__(self, rng: numpy.random.Generator,
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        self.builderFactory = ProbeBuilderFactory(fileReaderChooser, fileWriterChooser)
        self.repositoryItemFactory = ProbeRepositoryItemFactory(rng)
