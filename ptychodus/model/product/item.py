from __future__ import annotations
from abc import ABC, abstractmethod
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.product import Product
from ptychodus.api.visualize import Plot2D

from .metadata import MetadataRepositoryItem
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .productGeometry import ProductGeometry
from .productValidator import ProductValidator
from .scan import ScanRepositoryItem

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
        self._validator = validator
        self._costs = costs

        self._addParameterRepository(self._metadata, observe=True)
        self._addParameterRepository(self._scan, observe=True)
        self._addParameterRepository(self._geometry, observe=False)
        self._addParameterRepository(self._probe, observe=True)
        self._addParameterRepository(self._object, observe=True)

    def getName(self) -> str:
        return self._metadata.getName()

    def setName(self, name: str) -> None:
        self._metadata.setName(name)

    def getMetadata(self) -> MetadataRepositoryItem:
        return self._metadata

    def getScan(self) -> ScanRepositoryItem:
        return self._scan

    def getGeometry(self) -> ProductGeometry:
        return self._geometry

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
