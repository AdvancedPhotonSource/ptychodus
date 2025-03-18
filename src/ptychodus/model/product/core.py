import numpy

from ptychodus.api.object import ObjectFileReader, ObjectFileWriter
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, ProbeFileReader, ProbeFileWriter
from ptychodus.api.product import ProductFileReader, ProductFileWriter
from ptychodus.api.scan import ScanFileReader, ScanFileWriter
from ptychodus.api.settings import SettingsRegistry

from ..patterns import AssembledDiffractionDataset, PatternSizer
from .api import ObjectAPI, ProbeAPI, ProductAPI, ScanAPI
from .object import ObjectBuilderFactory, ObjectRepositoryItemFactory, ObjectSettings
from .object_repository import ObjectRepository
from .probe import ProbeBuilderFactory, ProbeRepositoryItemFactory, ProbeSettings
from .probe_repository import ProbeRepository
from .product_repository import ProductRepository
from .scan import ScanBuilderFactory, ScanRepositoryItemFactory, ScanSettings
from .scan_repository import ScanRepository
from .settings import ProductSettings


class ProductCore(Observer):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settingsRegistry: SettingsRegistry,
        patternSizer: PatternSizer,
        dataset: AssembledDiffractionDataset,
        scanFileReaderChooser: PluginChooser[ScanFileReader],
        scanFileWriterChooser: PluginChooser[ScanFileWriter],
        fresnelZonePlateChooser: PluginChooser[FresnelZonePlate],
        probeFileReaderChooser: PluginChooser[ProbeFileReader],
        probeFileWriterChooser: PluginChooser[ProbeFileWriter],
        objectFileReaderChooser: PluginChooser[ObjectFileReader],
        objectFileWriterChooser: PluginChooser[ObjectFileWriter],
        productFileReaderChooser: PluginChooser[ProductFileReader],
        productFileWriterChooser: PluginChooser[ProductFileWriter],
        reinitObservable: Observable,
    ) -> None:
        super().__init__()
        self.settings = ProductSettings(settingsRegistry)

        self._scanSettings = ScanSettings(settingsRegistry)
        self._scanBuilderFactory = ScanBuilderFactory(
            self._scanSettings, scanFileReaderChooser, scanFileWriterChooser
        )
        self._scanRepositoryItemFactory = ScanRepositoryItemFactory(
            rng, self._scanSettings, self._scanBuilderFactory
        )

        self._probeSettings = ProbeSettings(settingsRegistry)
        self._probeBuilderFactory = ProbeBuilderFactory(
            self._probeSettings,
            dataset,
            fresnelZonePlateChooser,
            probeFileReaderChooser,
            probeFileWriterChooser,
        )
        self._probeRepositoryItemFactory = ProbeRepositoryItemFactory(
            rng, self._probeSettings, self._probeBuilderFactory
        )

        self._objectSettings = ObjectSettings(settingsRegistry)
        self._objectBuilderFactory = ObjectBuilderFactory(
            rng, self._objectSettings, objectFileReaderChooser, objectFileWriterChooser
        )
        self._objectRepositoryItemFactory = ObjectRepositoryItemFactory(
            rng, self._objectSettings, self._objectBuilderFactory
        )

        self.productRepository = ProductRepository(
            self.settings,
            patternSizer,
            dataset,
            self._scanRepositoryItemFactory,
            self._probeRepositoryItemFactory,
            self._objectRepositoryItemFactory,
        )
        self.productAPI = ProductAPI(
            self.settings,
            self.productRepository,
            productFileReaderChooser,
            productFileWriterChooser,
        )
        self.scanRepository = ScanRepository(self.productRepository)
        self.scanAPI = ScanAPI(self._scanSettings, self.scanRepository, self._scanBuilderFactory)
        self.probeRepository = ProbeRepository(self.productRepository)
        self.probeAPI = ProbeAPI(
            self._probeSettings, self.probeRepository, self._probeBuilderFactory
        )
        self.objectRepository = ObjectRepository(self.productRepository)
        self.objectAPI = ObjectAPI(
            self._objectSettings, self.objectRepository, self._objectBuilderFactory
        )

        # TODO vvv refactor vvv
        productFileReaderChooser.synchronize_with_parameter(self.settings.file_type)
        productFileWriterChooser.set_current_plugin(self.settings.file_type.get_value())
        scanFileReaderChooser.synchronize_with_parameter(self._scanSettings.fileType)
        scanFileWriterChooser.set_current_plugin(self._scanSettings.fileType.get_value())
        probeFileReaderChooser.synchronize_with_parameter(self._probeSettings.fileType)
        probeFileWriterChooser.set_current_plugin(self._probeSettings.fileType.get_value())
        objectFileReaderChooser.synchronize_with_parameter(self._objectSettings.fileType)
        objectFileWriterChooser.set_current_plugin(self._objectSettings.fileType.get_value())
        # TODO ^^^^^^^^^^^^^^^^

        self._reinitObservable = reinitObservable
        reinitObservable.add_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self.productRepository.insertProductFromSettings()
