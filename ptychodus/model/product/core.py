from ...api.product import ProductFileReader, ProductFileWriter
from ...api.plugins import PluginChooser
from ..metadata import MetadataBuilder
from ..object import ObjectBuilderFactory, ObjectRepositoryItemFactory
from ..patterns import PatternSizer, ActiveDiffractionDataset
from ..probe import ProbeBuilderFactory, ProbeRepositoryItemFactory
from ..scan import ScanBuilderFactory, ScanRepositoryItemFactory
from .object import ObjectRepository
from .probe import ProbeRepository
from .repository import ProductRepository
from .scan import ScanRepository


class ProductCore:

    def __init__(self, patternSizer: PatternSizer, patterns: ActiveDiffractionDataset,
                 metadataBuilder: MetadataBuilder,
                 scanRepositoryItemFactory: ScanRepositoryItemFactory,
                 scanBuilderFactory: ScanBuilderFactory,
                 probeRepositoryItemFactory: ProbeRepositoryItemFactory,
                 probeBuilderFactory: ProbeBuilderFactory,
                 objectRepositoryItemFactory: ObjectRepositoryItemFactory,
                 objectBuilderFactory: ObjectBuilderFactory,
                 fileReaderChooser: PluginChooser[ProductFileReader],
                 fileWriterChooser: PluginChooser[ProductFileWriter]) -> None:
        self.repository = ProductRepository(patternSizer, patterns, metadataBuilder,
                                            scanRepositoryItemFactory, probeRepositoryItemFactory,
                                            objectRepositoryItemFactory, fileReaderChooser,
                                            fileWriterChooser)
        self.scanRepository = ScanRepository(self.repository, scanBuilderFactory)
        self.probeRepository = ProbeRepository(self.repository, probeBuilderFactory)
        self.objectRepository = ObjectRepository(self.repository, objectBuilderFactory)
