import logging

from ptychodus.api.product import Product

from ..patterns import AssembledDiffractionDataset, PatternSizer
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
    ) -> None:
        super().__init__()
        self._settings = settings
        self._pattern_sizer = pattern_sizer
        self._dataset = dataset
        self._scan_item_factory = scan_item_factory
        self._probe_item_factory = probe_item_factory
        self._object_item_factory = object_item_factory
        self._repository = repository

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
        mutable: bool = True,
        like_index: int = -1,
    ) -> ProductRepositoryItem:
        metadata_item = MetadataRepositoryItem(
            self._settings,
            name=name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
        )

        if mutable:
            scan_item = self._scan_item_factory.create()
        elif like_index >= 0:
            like_item = self._repository[like_index]
            scan_item = self._scan_item_factory.create(like_item.get_scan_item().get_scan())
        else:
            raise ValueError('FIXME1')  # FIXME

        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)

        if mutable:
            probe_item = self._probe_item_factory.create(geometry)
            object_item = self._object_item_factory.create(geometry)
        elif like_index >= 0:
            like_item = self._repository[like_index]
            probe_item = self._probe_item_factory.create(
                geometry, like_item.get_probe_item().get_probe()
            )
            object_item = self._object_item_factory.create(
                geometry, like_item.get_object_item().get_object()
            )
        else:
            raise ValueError('FIXME2')  # FIXME

        validator = ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item)

        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=validator,
            costs=list(),
        )

    def create_from_product(self, product: Product) -> ProductRepositoryItem:
        metadata_item = MetadataRepositoryItem(
            self._settings,
            name=product.metadata.name,
            comments=product.metadata.comments,
            detector_distance_m=product.metadata.detector_distance_m,
            probe_energy_eV=product.metadata.probe_energy_eV,
            probe_photon_count=product.metadata.probe_photon_count,
            exposure_time_s=product.metadata.exposure_time_s,
            mass_attenuation_m2_kg=product.metadata.mass_attenuation_m2_kg,
        )
        scan_item = self._scan_item_factory.create(product.positions)
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create(geometry, product.probe)
        object_item = self._object_item_factory.create(geometry, product.object_)

        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item),
            costs=product.costs,
        )

    def create_from_settings(self) -> ProductRepositoryItem:
        # TODO add mechanism to sync product state to settings
        # FIXME try to load from file, then fall back to load from components
        metadata_item = MetadataRepositoryItem(self._settings)
        scan_item = self._scan_item_factory.create_from_settings()
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create_from_settings(geometry)
        object_item = self._object_item_factory.create_from_settings(geometry)

        return ProductRepositoryItem(
            parent=self._repository,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item),
            costs=list(),
        )
