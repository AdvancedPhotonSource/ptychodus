import numpy

from ptychodus.api.object import ObjectFileReader, ObjectFileWriter
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter
from ptychodus.api.probe import ProbeFileReader, ProbeFileWriter
from ptychodus.api.probe_gen import FresnelZonePlate
from ptychodus.api.probe_positions import ProbePositionFileReader, ProbePositionFileWriter
from ptychodus.api.product import ProductFileReader, ProductFileWriter
from ptychodus.api.settings import SettingsRegistry

from ..diffraction import (
    AssembledDiffractionDataset,
    DiffractionAPI,
    DiffractionDatasetRepositoryObserver,
    PatternSizer,
)
from ..task_manager import TaskManager
from .api import ObjectAPI, ProbeAPI, ProductAPI, ProbePositionsAPI
from .item_factory import ProductRepositoryItemFactory
from .object import ObjectBuilderFactory, ObjectRepositoryItemFactory, ObjectSettings
from .object_repository import ObjectRepository
from .probe import ProbeBuilderFactory, ProbeRepositoryItemFactory, ProbeSettings
from .probe_repository import ProbeRepository
from .repository import ProductRepository
from .probe_positions import (
    ProbePositionsBuilderFactory,
    ProbePositionsRepositoryItemFactory,
    ProbePositionsSettings,
)
from .scan_repository import ProbePositionsRepository
from .settings import ProductSettings


class _DatasetOrphanObserver(DiffractionDatasetRepositoryObserver):
    """Clears product references to a diffraction dataset when it leaves the repository."""

    def __init__(self, product_repository: ProductRepository) -> None:
        self._product_repository = product_repository

    def handle_dataset_inserted(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        pass

    def handle_dataset_removed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        for item in self._product_repository:
            if item.get_dataset() is dataset:
                item.unbind_dataset()


class ProductCore(Observer):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings_registry: SettingsRegistry,
        pattern_sizer: PatternSizer,
        diffraction_api: DiffractionAPI,
        scan_file_reader_chooser: PluginChooser[ProbePositionFileReader],
        scan_file_writer_chooser: PluginChooser[ProbePositionFileWriter],
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
        probe_file_reader_chooser: PluginChooser[ProbeFileReader],
        probe_file_writer_chooser: PluginChooser[ProbeFileWriter],
        object_file_reader_chooser: PluginChooser[ObjectFileReader],
        object_file_writer_chooser: PluginChooser[ObjectFileWriter],
        product_file_reader_chooser: PluginChooser[ProductFileReader],
        product_file_writer_chooser: PluginChooser[ProductFileWriter],
        reinit_observable: Observable,
        task_manager: TaskManager,
    ) -> None:
        super().__init__()
        self.settings = ProductSettings(settings_registry)

        self._scan_settings = ProbePositionsSettings(settings_registry)
        self._scan_builder_factory = ProbePositionsBuilderFactory(
            rng, self._scan_settings, scan_file_reader_chooser, scan_file_writer_chooser
        )
        self._scan_repository_item_factory = ProbePositionsRepositoryItemFactory(
            rng, self._scan_settings, self._scan_builder_factory
        )

        self._probe_settings = ProbeSettings(settings_registry)
        self._probe_builder_factory = ProbeBuilderFactory(
            rng,
            self._probe_settings,
            fresnel_zone_plate_chooser,
            probe_file_reader_chooser,
            probe_file_writer_chooser,
        )
        self._probe_repository_item_factory = ProbeRepositoryItemFactory(
            rng, self._probe_settings, self._probe_builder_factory
        )

        self._object_settings = ObjectSettings(settings_registry)
        self._object_builder_factory = ObjectBuilderFactory(
            rng,
            self._object_settings,
            object_file_reader_chooser,
            object_file_writer_chooser,
        )
        self._object_repository_item_factory = ObjectRepositoryItemFactory(
            rng, self._object_settings, self._object_builder_factory
        )

        self.product_repository = ProductRepository()
        self._item_factory = ProductRepositoryItemFactory(
            self.settings,
            pattern_sizer,
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
            task_manager,
        )
        self._diffraction_api = diffraction_api
        self._dataset_orphan_observer = _DatasetOrphanObserver(self.product_repository)
        diffraction_api.get_repository().add_observer(self._dataset_orphan_observer)
        self.probe_positions_repository = ProbePositionsRepository(self.product_repository)
        self.probe_positions_api = ProbePositionsAPI(
            self._scan_settings, self.probe_positions_repository, self._scan_builder_factory
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
        self.product_file_reader_parameter = PluginChooserParameter(
            product_file_reader_chooser, self.settings.file_type
        )
        product_file_writer_chooser.set_current_plugin(self.settings.file_type.get_value())
        self.scan_file_reader_parameter = PluginChooserParameter(
            scan_file_reader_chooser, self._scan_settings.file_type
        )
        scan_file_writer_chooser.set_current_plugin(self._scan_settings.file_type.get_value())
        self.probe_file_reader_parameter = PluginChooserParameter(
            probe_file_reader_chooser, self._probe_settings.file_type
        )
        probe_file_writer_chooser.set_current_plugin(self._probe_settings.file_type.get_value())
        self.object_file_reader_parameter = PluginChooserParameter(
            object_file_reader_chooser, self._object_settings.file_type
        )
        object_file_writer_chooser.set_current_plugin(self._object_settings.file_type.get_value())
        # TODO ^^^^^^^^^^^^^^^^

        self._reinit_observable = reinit_observable
        reinit_observable.add_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._reinit_observable:
            # Depends on DiffractionCore being registered as a reinit observer
            # before ProductCore, so that open_patterns has already inserted the
            # settings-driven dataset into the repository by the time we run.
            repo = self._diffraction_api.get_repository()
            dataset = repo[-1] if len(repo) > 0 else None
            self.product_api.insert_product_from_settings(dataset=dataset, block=False)
