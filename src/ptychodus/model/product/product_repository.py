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
        patternSizer: PatternSizer,
        dataset: AssembledDiffractionDataset,
        scanRepositoryItemFactory: ScanRepositoryItemFactory,
        probeRepositoryItemFactory: ProbeRepositoryItemFactory,
        objectRepositoryItemFactory: ObjectRepositoryItemFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._patternSizer = patternSizer
        self._dataset = dataset
        self._scanRepositoryItemFactory = scanRepositoryItemFactory
        self._probeRepositoryItemFactory = probeRepositoryItemFactory
        self._objectRepositoryItemFactory = objectRepositoryItemFactory
        self._itemList: list[ProductRepositoryItem] = list()
        self._metadataRepositoryItemFactory = MetadataRepositoryItemFactory(self, settings)
        self._observerList: list[ProductRepositoryObserver] = [
            self._metadataRepositoryItemFactory,  # NOTE must be first!
        ]

    @overload
    def __getitem__(self, index: int) -> ProductRepositoryItem: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProductRepositoryItem]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ProductRepositoryItem | Sequence[ProductRepositoryItem]:
        return self._itemList[index]

    def __len__(self) -> int:
        return len(self._itemList)

    def _insertProduct(self, item: ProductRepositoryItem) -> int:
        index = len(self._itemList)
        self._itemList.append(item)

        for observer in self._observerList:
            observer.handle_item_inserted(index, item)

        return index

    def insert_new_product(
        self,
        *,
        name: str = '',
        comments: str = '',
        detectorDistanceInMeters: float | None = None,
        probeEnergyInElectronVolts: float | None = None,
        probePhotonCount: float | None = None,
        exposureTimeInSeconds: float | None = None,
        like_index: int,
    ) -> int:
        metadataItem = self._metadataRepositoryItemFactory.createDefault(
            name=name,
            comments=comments,
            detectorDistanceInMeters=detectorDistanceInMeters,
            probeEnergyInElectronVolts=probeEnergyInElectronVolts,
            probePhotonCount=probePhotonCount,
            exposureTimeInSeconds=exposureTimeInSeconds,
        )
        scanItem = self._scanRepositoryItemFactory.create()
        geometry = ProductGeometry(self._patternSizer, metadataItem, scanItem)
        probeItem = self._probeRepositoryItemFactory.create(geometry)
        objectItem = self._objectRepositoryItemFactory.create(geometry)

        item = ProductRepositoryItem(
            parent=self,
            metadata=metadataItem,
            scan=scanItem,
            geometry=geometry,
            probe=probeItem,
            object_=objectItem,
            validator=ProductValidator(self._dataset, scanItem, geometry, probeItem, objectItem),
            costs=list(),
        )

        index = self._insertProduct(item)

        if like_index >= 0:
            item.assign_item(self._itemList[like_index], notify=False)

        return index

    def insertProductFromSettings(self) -> int:
        # TODO add mechanism to sync product state to settings
        metadataItem = self._metadataRepositoryItemFactory.createDefault()
        scanItem = self._scanRepositoryItemFactory.createFromSettings()
        geometry = ProductGeometry(self._patternSizer, metadataItem, scanItem)
        probeItem = self._probeRepositoryItemFactory.createFromSettings(geometry)
        objectItem = self._objectRepositoryItemFactory.createFromSettings(geometry)

        item = ProductRepositoryItem(
            parent=self,
            metadata=metadataItem,
            scan=scanItem,
            geometry=geometry,
            probe=probeItem,
            object_=objectItem,
            validator=ProductValidator(self._dataset, scanItem, geometry, probeItem, objectItem),
            costs=list(),
        )

        return self._insertProduct(item)

    def insertProduct(self, product: Product) -> int:
        metadataItem = self._metadataRepositoryItemFactory.create(product.metadata)
        scanItem = self._scanRepositoryItemFactory.create(product.positions)
        geometry = ProductGeometry(self._patternSizer, metadataItem, scanItem)
        probeItem = self._probeRepositoryItemFactory.create(geometry, product.probe)
        objectItem = self._objectRepositoryItemFactory.create(geometry, product.object_)

        item = ProductRepositoryItem(
            parent=self,
            metadata=metadataItem,
            scan=scanItem,
            geometry=geometry,
            probe=probeItem,
            object_=objectItem,
            validator=ProductValidator(self._dataset, scanItem, geometry, probeItem, objectItem),
            costs=product.costs,
        )

        return self._insertProduct(item)

    def removeProduct(self, index: int) -> None:
        try:
            item = self._itemList.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove product item {index}!')
            return

        for observer in self._observerList:
            observer.handle_item_removed(index, item)

    def getInfoText(self) -> str:
        sizeInMB = sum(sys.getsizeof(prod) for prod in self._itemList) / (1024 * 1024)
        return f'Total: {len(self)} [{sizeInMB:.2f}MB]'

    def add_observer(self, observer: ProductRepositoryObserver) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: ProductRepositoryObserver) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def handle_metadata_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata()
        index = metadata.get_index()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observerList:
            observer.handle_metadata_changed(index, metadata)

    def handle_scan_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata()
        index = metadata.get_index()
        scan = item.get_scan()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observerList:
            observer.handle_scan_changed(index, scan)

    def handle_probe_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata()
        index = metadata.get_index()
        probe = item.get_probe()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observerList:
            observer.handle_probe_changed(index, probe)

    def handle_object_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata()
        index = metadata.get_index()
        object_ = item.get_object()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observerList:
            observer.handle_object_changed(index, object_)

    def handle_costs_changed(self, item: ProductRepositoryItem) -> None:
        metadata = item.get_metadata()
        index = metadata.get_index()
        costs = item.get_costs()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observerList:
            observer.handle_costs_changed(index, costs)
