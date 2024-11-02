from __future__ import annotations
from collections.abc import Sequence
from typing import overload
import logging
import sys

from ptychodus.api.product import Product

from ..patterns import ActiveDiffractionDataset, PatternSizer, ProductSettings
from .item import (
    ProductRepositoryItem,
    ProductRepositoryItemObserver,
    ProductRepositoryObserver,
)
from .metadataFactory import MetadataRepositoryItemFactory
from .object import ObjectRepositoryItemFactory
from .probe import ProbeRepositoryItemFactory
from .productGeometry import ProductGeometry
from .productValidator import ProductValidator
from .scan import ScanRepositoryItemFactory

logger = logging.getLogger(__name__)


class ProductRepository(Sequence[ProductRepositoryItem], ProductRepositoryItemObserver):
    def __init__(
        self,
        settings: ProductSettings,
        patternSizer: PatternSizer,
        patterns: ActiveDiffractionDataset,
        scanRepositoryItemFactory: ScanRepositoryItemFactory,
        probeRepositoryItemFactory: ProbeRepositoryItemFactory,
        objectRepositoryItemFactory: ObjectRepositoryItemFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._patternSizer = patternSizer
        self._patterns = patterns
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
            observer.handleItemInserted(index, item)

        return index

    def insertNewProduct(
        self,
        *,
        name: str = '',
        comments: str = '',
        detectorDistanceInMeters: float | None = None,
        probeEnergyInElectronVolts: float | None = None,
        probePhotonsPerSecond: float | None = None,
        exposureTimeInSeconds: float | None = None,
        likeIndex: int,
    ) -> int:
        metadataItem = self._metadataRepositoryItemFactory.createDefault(
            name=name,
            comments=comments,
            detectorDistanceInMeters=detectorDistanceInMeters,
            probeEnergyInElectronVolts=probeEnergyInElectronVolts,
            probePhotonsPerSecond=probePhotonsPerSecond,
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
            validator=ProductValidator(self._patterns, scanItem, geometry, probeItem, objectItem),
            costs=list(),
        )

        index = self._insertProduct(item)

        if likeIndex >= 0:
            item.assignItem(self._itemList[likeIndex], notify=False)

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
            validator=ProductValidator(self._patterns, scanItem, geometry, probeItem, objectItem),
            costs=list(),
        )

        return self._insertProduct(item)

    def insertProduct(self, product: Product) -> int:
        metadataItem = self._metadataRepositoryItemFactory.create(product.metadata)
        scanItem = self._scanRepositoryItemFactory.create(product.scan)
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
            validator=ProductValidator(self._patterns, scanItem, geometry, probeItem, objectItem),
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
            observer.handleItemRemoved(index, item)

    def getInfoText(self) -> str:
        sizeInMB = sum(sys.getsizeof(prod) for prod in self._itemList) / (1024 * 1024)
        return f'Total: {len(self)} [{sizeInMB:.2f}MB]'

    def addObserver(self, observer: ProductRepositoryObserver) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: ProductRepositoryObserver) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def handleMetadataChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.getName()}"!')
            return

        for observer in self._observerList:
            observer.handleMetadataChanged(index, metadata)

    def handleScanChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        scan = item.getScan()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.getName()}"!')
            return

        for observer in self._observerList:
            observer.handleScanChanged(index, scan)

    def handleProbeChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        probe = item.getProbe()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.getName()}"!')
            return

        for observer in self._observerList:
            observer.handleProbeChanged(index, probe)

    def handleObjectChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        object_ = item.getObject()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.getName()}"!')
            return

        for observer in self._observerList:
            observer.handleObjectChanged(index, object_)

    def handleCostsChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        costs = item.getCosts()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.getName()}"!')
            return

        for observer in self._observerList:
            observer.handleCostsChanged(index, costs)
