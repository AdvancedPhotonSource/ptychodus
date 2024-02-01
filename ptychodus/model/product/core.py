from ...api.product import ProductFileReader, ProductFileWriter
from ...api.plugins import PluginChooser
from ..metadata import MetadataBuilder
from ..object import ObjectBuilderFactory, ObjectRepositoryItemFactory
from ..patterns import PatternSizer, ActiveDiffractionDataset
from ..probe import ProbeBuilderFactory, ProbeRepositoryItemFactory
from ..scan import ScanBuilderFactory, ScanRepositoryItemFactory
from .metadata import MetadataRepository
from .object import ObjectRepository
from .probe import ProbeRepository
from .repository import ProductRepository
from .scan import ScanRepository


class ProductCore:

    def __init__(self, patternSizer: PatternSizer, patterns: ActiveDiffractionDataset,
                 metadataBuilder: MetadataBuilder,
                 scanRepositoryItemFactory: ScanRepositoryItemFactory,
                 scanBuilderFactory: ScanBuilderFactory,
                 objectRepositoryItemFactory: ObjectRepositoryItemFactory,
                 objectBuilderFactory: ObjectBuilderFactory,
                 probeRepositoryItemFactory: ProbeRepositoryItemFactory,
                 probeBuilderFactory: ProbeBuilderFactory,
                 fileReaderChooser: PluginChooser[ProductFileReader],
                 fileWriterChooser: PluginChooser[ProductFileWriter]) -> None:
        self._repository = ProductRepository(patternSizer, patterns, scanRepositoryItemFactory,
                                             probeRepositoryItemFactory,
                                             objectRepositoryItemFactory)
        self.metadataRepository = MetadataRepository(self._repository, metadataBuilder,
                                                     fileReaderChooser, fileWriterChooser)
        self.scanRepository = ScanRepository(self._repository, scanBuilderFactory)
        self.probeRepository = ProbeRepository(self._repository, probeBuilderFactory)
        self.objectRepository = ObjectRepository(self._repository, objectBuilderFactory)
