from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.product import Product

from .metadata import MetadataRepositoryItem
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .product_geometry import ProductGeometry
from .product_validator import ProductValidator
from .scan import ScanRepositoryItem

logger = logging.getLogger(__name__)


class ProductRepositoryItemObserver(ABC):
    @abstractmethod
    def handle_metadata_changed(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_scan_changed(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_probe_changed(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_object_changed(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_costs_changed(self, item: ProductRepositoryItem) -> None:
        pass


class ProductRepositoryItem(ParameterGroup):
    def __init__(
        self,
        parent: ProductRepositoryItemObserver,
        metadata: MetadataRepositoryItem,
        scan: ScanRepositoryItem,
        geometry: ProductGeometry,
        probe: ProbeRepositoryItem,
        object_: ObjectRepositoryItem,
        validator: ProductValidator,
        costs: Sequence[float],
    ) -> None:
        super().__init__()
        self._parent = parent
        self._metadata = metadata
        self._scan = scan
        self._geometry = geometry
        self._probe = probe
        self._object = object_
        self._validator = validator
        self._costs = list(costs)

        self._add_group('metadata', self._metadata, observe=True)
        self._add_group('scan', self._scan, observe=True)
        self._add_group('probe', self._probe, observe=True)
        self._add_group('object', self._object, observe=True)

    def assign_item(self, item: ProductRepositoryItem, *, notify: bool = True) -> None:
        self._metadata.assign_item(item.get_metadata())
        self._scan.assign_item(item.get_scan())
        self._probe.assign_item(item.get_probe())
        self._object.assign_item(item.get_object())
        self._costs = list(item.get_costs())

        if notify:
            self._parent.handle_costs_changed(self)

    def assign(self, product: Product, *, mutable: bool = True) -> None:
        self._metadata.assign(product.metadata)
        self._scan.assign(product.positions, mutable=mutable)
        self._probe.assign(product.probe, mutable=mutable)
        self._object.assign(product.object_, mutable=mutable)
        self._costs = list(product.costs)
        self._parent.handle_costs_changed(self)

    def sync_to_settings(self) -> None:
        self._metadata.sync_to_settings()
        self._scan.sync_to_settings()
        self._probe.sync_to_settings()
        self._object.sync_to_settings()

    def get_name(self) -> str:
        return self._metadata.get_name()

    def set_name(self, name: str) -> None:
        self._metadata.set_name(name)

    def get_metadata(self) -> MetadataRepositoryItem:
        return self._metadata

    def get_scan(self) -> ScanRepositoryItem:
        return self._scan

    def get_geometry(self) -> ProductGeometry:
        return self._geometry

    def get_probe(self) -> ProbeRepositoryItem:
        return self._probe

    def get_object(self) -> ObjectRepositoryItem:
        return self._object

    def _invalidate_costs(self) -> None:
        self._costs = list()
        self._parent.handle_costs_changed(self)

    def get_costs(self) -> Sequence[float]:
        return self._costs

    def get_product(self) -> Product:
        return Product(
            metadata=self._metadata.get_metadata(),
            positions=self._scan.getScan(),
            probe=self._probe.get_probe(),
            object_=self._object.get_object(),
            costs=self.get_costs(),
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self._invalidate_costs()
            self._parent.handle_metadata_changed(self)
        elif observable is self._scan:
            self._invalidate_costs()
            self._parent.handle_scan_changed(self)
        elif observable is self._probe:
            self._invalidate_costs()
            self._parent.handle_probe_changed(self)
        elif observable is self._object:
            self._invalidate_costs()
            self._parent.handle_object_changed(self)


class ProductRepositoryObserver(ABC):
    @abstractmethod
    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_scan_changed(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_costs_changed(self, index: int, costs: Sequence[float]) -> None:
        pass

    @abstractmethod
    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        pass
