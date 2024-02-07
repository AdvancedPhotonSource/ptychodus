from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import overload
import logging

from ...api.observer import Observable
from ...api.parametric import ParameterRepository
from ...api.product import Product
from ...api.visualize import Plot2D
from ..metadata import MetadataRepositoryItem
from ..object import ObjectRepositoryItem, ObjectRepositoryItemFactory
from ..patterns import ActiveDiffractionDataset, PatternSizer
from ..probe import ProbeRepositoryItem, ProbeRepositoryItemFactory
from ..scan import ScanRepositoryItem, ScanRepositoryItemFactory
from .geometry import ProductGeometry
from .validator import ProductValidator

logger = logging.getLogger(__name__)


class ProductRepositoryItemObserver(ABC):

    @abstractmethod
    def handleMetadataChanged(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleScanChanged(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleProbeChanged(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleObjectChanged(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleCostsChanged(self, item: ProductRepositoryItem) -> None:
        pass


class ProductRepositoryItem(ParameterRepository):

    def __init__(self, parent: ProductRepositoryItemObserver, metadata: MetadataRepositoryItem,
                 scan: ScanRepositoryItem, geometry: ProductGeometry, probe: ProbeRepositoryItem,
                 object_: ObjectRepositoryItem, validator: ProductValidator,
                 costs: Plot2D) -> None:
        super().__init__('Product')
        self._parent = parent
        self._metadata = metadata
        self._scan = scan
        self._geometry = geometry
        self._probe = probe
        self._object = object_
        self._costs = costs

        self._addParameterRepository(self._metadata, observe=True)
        self._addParameterRepository(self._scan, observe=True)
        self._addParameterRepository(self._geometry, observe=False)
        self._addParameterRepository(self._probe, observe=True)
        self._addParameterRepository(self._object, observe=True)

    @property
    def name(self) -> str:
        return self._metadata.name.getValue()

    def getMetadata(self) -> MetadataRepositoryItem:
        return self._metadata

    def getScan(self) -> ScanRepositoryItem:
        return self._scan

    def getProbe(self) -> ProbeRepositoryItem:
        return self._probe

    def getObject(self) -> ObjectRepositoryItem:
        return self._object

    def _invalidateCosts(self) -> None:
        self._costs = Plot2D.createNull()
        self._parent.handleCostsChanged(self)

    def getCosts(self) -> Plot2D:
        return self._costs

    def getProduct(self) -> Product:
        return Product(
            metadata=self._metadata.getMetadata(),
            scan=self._scan.getScan(),
            probe=self._probe.getProbe(),
            object_=self._object.getObject(),
            costs=self.getCosts(),
        )

    def update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self._invalidateCosts()
            self._parent.handleMetadataChanged(self)
        elif observable is self._scan:
            self._invalidateCosts()
            self._parent.handleScanChanged(self)
        elif observable is self._probe:
            self._invalidateCosts()
            self._parent.handleProbeChanged(self)
        elif observable is self._object:
            self._invalidateCosts()
            self._parent.handleObjectChanged(self)


class ProductRepositoryObserver(ABC):

    @abstractmethod
    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleCostsChanged(self, index: int, costs: Plot2D) -> None:
        pass

    @abstractmethod
    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        pass


class ProductRepository(Sequence[ProductRepositoryItem], ProductRepositoryItemObserver):

    def __init__(self, patternSizer: PatternSizer, patterns: ActiveDiffractionDataset,
                 scanRepositoryItemFactory: ScanRepositoryItemFactory,
                 probeRepositoryItemFactory: ProbeRepositoryItemFactory,
                 objectRepositoryItemFactory: ObjectRepositoryItemFactory) -> None:
        super().__init__()
        self._patternSizer = patternSizer
        self._patterns = patterns
        self._scanRepositoryItemFactory = scanRepositoryItemFactory
        self._probeRepositoryItemFactory = probeRepositoryItemFactory
        self._objectRepositoryItemFactory = objectRepositoryItemFactory
        self._itemIndexLut: dict[str, int] = dict()
        self._itemList: list[ProductRepositoryItem] = list()
        self._observerList: list[ProductRepositoryObserver] = list()

    @overload
    def __getitem__(self, index: int) -> ProductRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProductRepositoryItem]:
        ...

    def __getitem__(self,
                    index: int | slice) -> ProductRepositoryItem | Sequence[ProductRepositoryItem]:
        return self._itemList[index]

    def __len__(self) -> int:
        return len(self._itemList)

    def insertProduct(self, product: Product) -> int:
        index = len(self._itemList)
        metadata = MetadataRepositoryItem(product.metadata)
        name = metadata.name.getValue()
        match = 0

        while self._itemIndexLUT.setdefault(name, index) != index:
            match += 1
            name = metadata.name.getValue() + f'-{match}'

        metadata.name.setValue(name)

        scan = self._scanRepositoryItemFactory.create(product.scan)
        geometry = ProductGeometry(metadata, scan, self._patternSizer)
        probe = self._probeRepositoryItemFactory.create(product.probe)
        object_ = self._objectRepositoryItemFactory.create(product.object_)
        item = ProductRepositoryItem(
            parent=self,
            metadata=metadata,
            scan=scan,
            geometry=geometry,
            probe=probe,
            object_=object_,
            validator=ProductValidator(self._patterns, scan, geometry, probe, object_),
            costs=product.costs,
        )
        self._itemList.append(item)

        for observer in self._observerList:
            observer.handleItemInserted(index, item)

        return index

    def _updateIndexes(self) -> None:
        self._itemIndexLUT = {item.name: index for index, item in enumerate(self._itemList)}

    def removeProduct(self, index: int) -> None:
        try:
            item = self._itemList.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove product item {index}!')
            return

        self._updateIndexes()

        for observer in self._observerList:
            observer.handleItemRemoved(index, item)

    def addObserver(self, observer: ProductRepositoryObserver) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: ProductRepositoryObserver) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def handleMetadataChanged(self, item: ProductRepositoryItem) -> None:
        try:
            index = self._itemIndexLUT[item.name]
        except KeyError:
            logger.warning(f'Failed to look up index for \"{item.name}\"!')
            return

        metadata = item.getMetadata()
        self._updateIndexes()  # FIXME also enforce that name remains unique

        for observer in self._observerList:
            observer.handleMetadataChanged(index, metadata)

    def handleScanChanged(self, item: ProductRepositoryItem) -> None:
        try:
            index = self._itemIndexLUT[item.name]
        except KeyError:
            logger.warning(f'Failed to look up index for \"{item.name}\"!')
            return

        scan = item.getScan()

        for observer in self._observerList:
            observer.handleScanChanged(index, scan)

    def handleProbeChanged(self, item: ProductRepositoryItem) -> None:
        try:
            index = self._itemIndexLUT[item.name]
        except KeyError:
            logger.warning(f'Failed to look up index for \"{item.name}\"!')
            return

        probe = item.getProbe()

        for observer in self._observerList:
            observer.handleProbeChanged(index, probe)

    def handleObjectChanged(self, item: ProductRepositoryItem) -> None:
        try:
            index = self._itemIndexLUT[item.name]
        except KeyError:
            logger.warning(f'Failed to look up index for \"{item.name}\"!')
            return

        object_ = item.getObject()

        for observer in self._observerList:
            observer.handleObjectChanged(index, object_)

    def handleCostsChanged(self, item: ProductRepositoryItem) -> None:
        try:
            index = self._itemIndexLUT[item.name]
        except KeyError:
            logger.warning(f'Failed to look up index for \"{item.name}\"!')
            return

        costs = item.getCosts()

        for observer in self._observerList:
            observer.handleCostsChanged(index, costs)
