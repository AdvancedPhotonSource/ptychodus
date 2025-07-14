import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product, ProductFileReader

from ..diffraction import AssembledDiffractionDataset, PatternSizer
from .geometry import ProductGeometry
from .item import ProductRepositoryItem
from .metadata import MetadataRepositoryItem
from .object import ObjectRepositoryItemFactory
from .probe import ProbeRepositoryItemFactory
from .repository import ProductRepository
from .scan import ScanRepositoryItemFactory
from .settings import ProductSettings
from .validator import ProductValidator

logger = logging.getLogger(__name__)


class ProductRepositoryItemFactory:
    def __init__(
        self,
        settings: ProductSettings,
        pattern_sizer: PatternSizer,
        dataset: AssembledDiffractionDataset,
        scan_item_factory: ScanRepositoryItemFactory,
        probe_item_factory: ProbeRepositoryItemFactory,
        object_item_factory: ObjectRepositoryItemFactory,
        repository: ProductRepository,
        file_reader_chooser: PluginChooser[ProductFileReader],
    ) -> None:
        super().__init__()
        self._settings = settings
        self._pattern_sizer = pattern_sizer
        self._dataset = dataset
        self._scan_item_factory = scan_item_factory
        self._probe_item_factory = probe_item_factory
        self._object_item_factory = object_item_factory
        self._repository = repository
        self._file_reader_chooser = file_reader_chooser

    def create_from_values(
        self,
        *,
        name: str = '',
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mass_attenuation_m2_kg: float | None = None,
        tomography_angle_deg: float | None = None,
    ) -> ProductRepositoryItem:
        metadata_item = MetadataRepositoryItem(
            self._settings,
            self._repository,
            name=name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
            tomography_angle_deg=tomography_angle_deg,
        )

        if metadata_item.probe_photon_count.get_value() <= 0:
            metadata_item.probe_photon_count.set_value(self._dataset.get_maximum_pattern_counts())

        scan_item = self._scan_item_factory.create()
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create(geometry)
        object_item = self._object_item_factory.create(geometry)
        validator = ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item)

        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=validator,
            losses=list(),
        )

    def create_from_product(self, product: Product) -> ProductRepositoryItem:
        metadata_item = MetadataRepositoryItem(
            self._settings,
            self._repository,
            name=product.metadata.name,
            comments=product.metadata.comments,
            detector_distance_m=product.metadata.detector_distance_m,
            probe_energy_eV=product.metadata.probe_energy_eV,
            probe_photon_count=product.metadata.probe_photon_count,
            exposure_time_s=product.metadata.exposure_time_s,
            mass_attenuation_m2_kg=product.metadata.mass_attenuation_m2_kg,
            tomography_angle_deg=product.metadata.tomography_angle_deg,
        )

        scan_item = self._scan_item_factory.create(product.positions)
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create(geometry, product.probes)
        object_item = self._object_item_factory.create(geometry, product.object_)
        validator = ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item)

        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=validator,
            losses=product.losses,
        )

    def create_from_settings(self) -> ProductRepositoryItem:
        file_path = self._settings.file_path.get_value()

        if file_path.is_file():
            file_type = self._file_reader_chooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{file_path}" as "{file_type}"')
            file_reader = self._file_reader_chooser.get_current_plugin().strategy

            try:
                product = file_reader.read(file_path)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{file_path}"') from exc
            else:
                return self.create_from_product(product)

        metadata_item = MetadataRepositoryItem(self._settings, self._repository)
        scan_item = self._scan_item_factory.create_from_settings()
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create_from_settings(geometry)
        object_item = self._object_item_factory.create_from_settings(geometry)

        item = ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item),
            losses=list(),
        )
        logger.debug(f'Created product from settings: {item.get_name()}')
        return item
