import numpy

from ptychodus.api.object import ObjectFileReader, ObjectFileWriter
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, ProbeFileReader, ProbeFileWriter
from ptychodus.api.product import ProductFileReader, ProductFileWriter
from ptychodus.api.scan import PositionFileReader, PositionFileWriter
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import AssembledDiffractionDataset, PatternSizer
from .api import ObjectAPI, ProbeAPI, ProductAPI, ScanAPI
from .item_factory import ProductRepositoryItemFactory
from .object import ObjectBuilderFactory, ObjectRepositoryItemFactory, ObjectSettings
from .object_repository import ObjectRepository
from .probe import ProbeBuilderFactory, ProbeRepositoryItemFactory, ProbeSettings
from .probe_repository import ProbeRepository
from .repository import ProductRepository
from .scan import ScanBuilderFactory, ScanRepositoryItemFactory, ScanSettings
from .scan_repository import ScanRepository
from .settings import ProductSettings


class ProductCore(Observer):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings_registry: SettingsRegistry,
        pattern_sizer: PatternSizer,
        dataset: AssembledDiffractionDataset,
        scan_file_reader_chooser: PluginChooser[PositionFileReader],
        scan_file_writer_chooser: PluginChooser[PositionFileWriter],
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
        probe_file_reader_chooser: PluginChooser[ProbeFileReader],
        probe_file_writer_chooser: PluginChooser[ProbeFileWriter],
        object_file_reader_chooser: PluginChooser[ObjectFileReader],
        object_file_writer_chooser: PluginChooser[ObjectFileWriter],
        product_file_reader_chooser: PluginChooser[ProductFileReader],
        product_file_writer_chooser: PluginChooser[ProductFileWriter],
        reinit_observable: Observable,
    ) -> None:
        super().__init__()
        self.settings = ProductSettings(settings_registry)

        self._scan_settings = ScanSettings(settings_registry)
        self._scan_builder_factory = ScanBuilderFactory(
            self._scan_settings, scan_file_reader_chooser, scan_file_writer_chooser
        )
        self._scan_repository_item_factory = ScanRepositoryItemFactory(
            rng, self._scan_settings, self._scan_builder_factory
        )

        self._probe_settings = ProbeSettings(settings_registry)
        self._probe_builder_factory = ProbeBuilderFactory(
            self._probe_settings,
            dataset,
            fresnel_zone_plate_chooser,
            probe_file_reader_chooser,
            probe_file_writer_chooser,
        )
        self._probe_repository_item_factory = ProbeRepositoryItemFactory(
            rng, self._probe_settings, self._probe_builder_factory
        )

        self._object_settings = ObjectSettings(settings_registry)
        self._object_builder_factory = ObjectBuilderFactory(
            rng, self._object_settings, object_file_reader_chooser, object_file_writer_chooser
        )
        self._object_repository_item_factory = ObjectRepositoryItemFactory(
            rng, self._object_settings, self._object_builder_factory
        )

        self.product_repository = ProductRepository()
        self._item_factory = ProductRepositoryItemFactory(
            self.settings,
            pattern_sizer,
            dataset,
            self._scan_repository_item_factory,
            self._probe_repository_item_factory,
            self._object_repository_item_factory,
            self.product_repository,
            product_file_reader_chooser,
        )
        self.product_api = ProductAPI(
            self.settings,
            self.product_repository,
            self._item_factory,
            product_file_reader_chooser,
            product_file_writer_chooser,
        )
        self.scan_repository = ScanRepository(self.product_repository)
        self.scan_api = ScanAPI(
            self._scan_settings, self.scan_repository, self._scan_builder_factory
        )
        self.probe_repository = ProbeRepository(self.product_repository)
        self.probe_api = ProbeAPI(
            self._probe_settings, self.probe_repository, self._probe_builder_factory
        )
        self.object_repository = ObjectRepository(self.product_repository)
        self.object_api = ObjectAPI(
            self._object_settings, self.object_repository, self._object_builder_factory
        )

        # TODO vvv refactor vvv
        product_file_reader_chooser.synchronize_with_parameter(self.settings.file_type)
        product_file_writer_chooser.set_current_plugin(self.settings.file_type.get_value())
        scan_file_reader_chooser.synchronize_with_parameter(self._scan_settings.file_type)
        scan_file_writer_chooser.set_current_plugin(self._scan_settings.file_type.get_value())
        probe_file_reader_chooser.synchronize_with_parameter(self._probe_settings.file_type)
        probe_file_writer_chooser.set_current_plugin(self._probe_settings.file_type.get_value())
        object_file_reader_chooser.synchronize_with_parameter(self._object_settings.file_type)
        object_file_writer_chooser.set_current_plugin(self._object_settings.file_type.get_value())
        # TODO ^^^^^^^^^^^^^^^^

        self._reinit_observable = reinit_observable
        reinit_observable.add_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._reinit_observable:
            self.product_api.insert_product_from_settings()
