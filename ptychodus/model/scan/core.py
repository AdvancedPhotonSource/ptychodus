import numpy

from ...api.plugins import PluginChooser
from ...api.scan import ScanFileReader, ScanFileWriter
from .builderFactory import ScanBuilderFactory
from .itemFactory import ScanRepositoryItemFactory
from .transform import ScanPointTransformFactory


class ScanCore:

    def __init__(self, rng: numpy.random.Generator,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self.builderFactory = ScanBuilderFactory(fileReaderChooser, fileWriterChooser)
        self.transformFactory = ScanPointTransformFactory(rng)
        self.repositoryItemFactory = ScanRepositoryItemFactory(self.transformFactory)
