from __future__ import annotations
from collections.abc import Sequence
from typing import overload
import logging
import sys

from ptychodus.api.product import Product

from ..patterns import AssembledDiffractionDataset, PatternSizer
from .item import (
    ProductRepositoryItem,
    ProductRepositoryItemObserver,
    ProductRepositoryObserver,
)
from .metadata_factory import MetadataRepositoryItemFactory
from .object import ObjectRepositoryItemFactory
from .probe import ProbeRepositoryItemFactory
from .product_geometry import ProductGeometry
from .product_validator import ProductValidator
from .scan import ScanRepositoryItemFactory
from .settings import ProductSettings

logger = logging.getLogger(__name__)


class ProductRepository(Sequence[ProductRepositoryItem], ProductRepositoryItemObserver):
    def __init__(
        self,
        settings: ProductSettings,
        pattern_sizer: PatternSizer,
        dataset: AssembledDiffractionDataset,
        scan_item_factory: ScanRepositoryItemFactory,
        probe_item_factory: ProbeRepositoryItemFactory,
        object_item_factory: ObjectRepositoryItemFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._pattern_sizer = pattern_sizer
        self._dataset = dataset
        self._scan_item_factory = scan_item_factory
        self._probe_item_factory = probe_item_factory
        self._object_item_factory = object_item_factory
        self._item_list: list[ProductRepositoryItem] = list()
        self._metadata_item_factory = MetadataRepositoryItemFactory(self, settings)
        self._observer_list: list[ProductRepositoryObserver] = [
            self._metadata_item_factory,  # NOTE must be first!
        ]

    @overload
    def __getitem__(self, index: int) -> ProductRepositoryItem: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProductRepositoryItem]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ProductRepositoryItem | Sequence[ProductRepositoryItem]:
        return self._item_list[index]

    def __len__(self) -> int:
        return len(self._item_list)

    def _insert_product(self, item: ProductRepositoryItem) -> int:
        index = len(self._item_list)
        self._item_list.append(item)

        for observer in self._observer_list:
            observer.handle_item_inserted(index, item)

        return index

    def insert_new_product(
        self,
        *,
        name: str = '',
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mutable: bool = True,
        like_index: int,
    ) -> int:
        like_item = self._item_list[like_index]
        metadata_item = self._metadata_item_factory.create_default(
            name=name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
        )

        if mutable:
            scan_item = self._scan_item_factory.create()
        else:
            scan_item = self._scan_item_factory.create(like_item.get_scan_item().get_scan())

        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)

        if mutable:
            probe_item = self._probe_item_factory.create(geometry)
            object_item = self._object_item_factory.create(geometry)
        else:
            probe_item = self._probe_item_factory.create(
                geometry, like_item.get_probe_item().get_probe()
            )
            object_item = self._object_item_factory.create(
                geometry, like_item.get_object_item().get_object()
            )

        validator = ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item)

        item = ProductRepositoryItem(
            parent=self,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=validator,
            costs=list(),
        )
        return self._insert_product(item)

    def insert_product_from_settings(self) -> int:
        # TODO add mechanism to sync product state to settings
        metadata_item = self._metadata_item_factory.create_default()
        scan_item = self._scan_item_factory.create_from_settings()
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create_from_settings(geometry)
        object_item = self._object_item_factory.create_from_settings(geometry)

        item = ProductRepositoryItem(
            parent=self,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item),
            costs=list(),
        )

        return self._insert_product(item)

    def insert_product(self, product: Product) -> int:
        metadata_item = self._metadata_item_factory.create(product.metadata)
        scan_item = self._scan_item_factory.create(product.positions)
        geometry = ProductGeometry(self._pattern_sizer, metadata_item, scan_item)
        probe_item = self._probe_item_factory.create(geometry, product.probe)
        object_item = self._object_item_factory.create(geometry, product.object_)

        item = ProductRepositoryItem(
            parent=self,
            metadata_item=metadata_item,
            scan_item=scan_item,
            geometry=geometry,
            probe_item=probe_item,
            object_item=object_item,
            validator=ProductValidator(self._dataset, scan_item, geometry, probe_item, object_item),
            costs=product.costs,
        )

        return self._insert_product(item)

    def remove_product(self, index: int) -> None:
        try:
            item = self._item_list.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove product item {index}!')
            return

        for observer in self._observer_list:
            observer.handle_item_removed(index, item)

    def get_info_text(self) -> str:
        size_MB = sum(sys.getsizeof(prod) for prod in self._item_list) / (1024 * 1024)  # noqa: N806
        return f'Total: {len(self)} [{size_MB:.2f}MB]'

    def add_observer(self, observer: ProductRepositoryObserver) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: ProductRepositoryObserver) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def handle_metadata_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata_item()
        index = metadata.get_index()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_metadata_changed(index, metadata)

    def handle_scan_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata_item()
        index = metadata.get_index()
        scan = item.get_scan_item()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_scan_changed(index, scan)

    def handle_probe_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata_item()
        index = metadata.get_index()
        probe = item.get_probe_item()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_probe_changed(index, probe)

    def handle_object_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata_item()
        index = metadata.get_index()
        object_ = item.get_object_item()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_object_changed(index, object_)

    def handle_costs_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata_item()
        index = metadata.get_index()
        costs = item.get_costs()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_costs_changed(index, costs)
