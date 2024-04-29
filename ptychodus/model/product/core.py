import numpy

from ptychodus.api.object import ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, ProbeFileReader, ProbeFileWriter
from ptychodus.api.product import ProductFileReader, ProductFileWriter
from ptychodus.api.scan import ScanFileReader, ScanFileWriter
from ptychodus.api.settings import SettingsRegistry

from ..patterns import ActiveDiffractionDataset, Detector, PatternSizer, ProductSettings
from .api import ObjectAPI, ProbeAPI, ProductAPI, ScanAPI
from .object import ObjectBuilderFactory, ObjectRepositoryItemFactory, ObjectSettings
from .objectRepository import ObjectRepository
from .probe import ProbeBuilderFactory, ProbeRepositoryItemFactory, ProbeSettings
from .probeRepository import ProbeRepository
from .productRepository import ProductRepository
from .scan import ScanBuilderFactory, ScanRepositoryItemFactory, ScanSettings
from .scanRepository import ScanRepository


class ProductCore:

    def __init__(
        self,
        rng: numpy.random.Generator,
        settingsRegistry: SettingsRegistry,
        detector: Detector,
        settings: ProductSettings,
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
        self._scanSettings = ScanSettings(settingsRegistry)
        self._scanBuilderFactory = ScanBuilderFactory(self._scanSettings, scanFileReaderChooser,
                                                      scanFileWriterChooser)
        self._scanRepositoryItemFactory = ScanRepositoryItemFactory(rng, self._scanSettings,
                                                                    self._scanBuilderFactory)

        self._probeSettings = ProbeSettings(settingsRegistry)
        self._probeBuilderFactory = ProbeBuilderFactory(self._probeSettings, detector, patterns,
                                                        fresnelZonePlateChooser,
                                                        probeFileReaderChooser,
                                                        probeFileWriterChooser)
        self._probeRepositoryItemFactory = ProbeRepositoryItemFactory(
            rng, self._probeSettings, self._probeBuilderFactory)

        self._objectSettings = ObjectSettings(settingsRegistry)
        self._objectBuilderFactory = ObjectBuilderFactory(rng, self._objectSettings,
                                                          objectFileReaderChooser,
                                                          objectFileWriterChooser)
        self._objectRepositoryItemFactory = ObjectRepositoryItemFactory(
            rng, self._objectSettings, self._objectBuilderFactory)

        self.productRepository = ProductRepository(settings, patternSizer, patterns,
                                                   self._scanRepositoryItemFactory,
                                                   self._probeRepositoryItemFactory,
                                                   self._objectRepositoryItemFactory)
        self.productAPI = ProductAPI(self.productRepository, productFileReaderChooser,
                                     productFileWriterChooser)
        self.scanRepository = ScanRepository(self.productRepository)
        self.scanAPI = ScanAPI(self.scanRepository, self._scanBuilderFactory)
        self.probeRepository = ProbeRepository(self.productRepository)
        self.probeAPI = ProbeAPI(self.probeRepository, self._probeBuilderFactory)
        self.objectRepository = ObjectRepository(self.productRepository)
        self.objectAPI = ObjectAPI(self.objectRepository, self._objectBuilderFactory)
