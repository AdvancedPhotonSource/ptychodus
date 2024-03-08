import numpy

from ptychodus.api.object import ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, ProbeFileReader, ProbeFileWriter
from ptychodus.api.product import ProductFileReader, ProductFileWriter
from ptychodus.api.scan import ScanFileReader, ScanFileWriter

from ..patterns import ActiveDiffractionDataset, DiffractionDatasetSettings, PatternSizer
from .object import ObjectBuilderFactory, ObjectRepositoryItemFactory
from .objectRepository import ObjectRepository
from .probe import ProbeBuilderFactory, ProbeRepositoryItemFactory
from .probeRepository import ProbeRepository
from .productRepository import ProductRepository
from .scan import ScanBuilderFactory, ScanRepositoryItemFactory
from .scanRepository import ScanRepository


class ProductCore:

    def __init__(
        self,
        rng: numpy.random.Generator,
        datasetSettings: DiffractionDatasetSettings,
        patternSizer: PatternSizer,
        patterns: ActiveDiffractionDataset,
        scanFileReaderChooser: PluginChooser[ScanFileReader],
        scanFileWriterChooser: PluginChooser[ScanFileWriter],
        fresnelZonePlateChooser: PluginChooser[FresnelZonePlate],
        probeFileReaderChooser: PluginChooser[ProbeFileReader],
        probeFileWriterChooser: PluginChooser[ProbeFileWriter],
        objectFileReaderChooser: PluginChooser[ObjectFileReader],
        objectFileWriterChooser: PluginChooser[ObjectFileWriter],
        productFileReaderChooser: PluginChooser[ProductFileReader],
        productFileWriterChooser: PluginChooser[ProductFileWriter],
    ) -> None:
        self._scanBuilderFactory = ScanBuilderFactory(scanFileReaderChooser, scanFileWriterChooser)
        self._scanRepositoryItemFactory = ScanRepositoryItemFactory(rng)
        self._probeBuilderFactory = ProbeBuilderFactory(fresnelZonePlateChooser,
                                                        probeFileReaderChooser,
                                                        probeFileWriterChooser)
        self._probeRepositoryItemFactory = ProbeRepositoryItemFactory(rng)
        self._objectBuilderFactory = ObjectBuilderFactory(rng, objectFileReaderChooser,
                                                          objectFileWriterChooser)
        self._objectRepositoryItemFactory = ObjectRepositoryItemFactory(rng)

        self.productRepository = ProductRepository(datasetSettings, patternSizer, patterns,
                                                   self._scanRepositoryItemFactory,
                                                   self._probeRepositoryItemFactory,
                                                   self._objectRepositoryItemFactory,
                                                   productFileReaderChooser,
                                                   productFileWriterChooser)
        self.scanRepository = ScanRepository(self.productRepository, self._scanBuilderFactory)
        self.probeRepository = ProbeRepository(self.productRepository, self._probeBuilderFactory)
        self.objectRepository = ObjectRepository(self.productRepository,
                                                 self._objectBuilderFactory)
