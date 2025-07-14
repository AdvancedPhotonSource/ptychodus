from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.product import LossValue, Product

from .geometry import ProductGeometry
from .metadata import MetadataRepositoryItem, UniqueNameFactory
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .scan import ScanRepositoryItem
from .validator import ProductValidator

logger = logging.getLogger(__name__)


class ProductRepositoryItemObserver(UniqueNameFactory):
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
    def handle_losses_changed(self, item: ProductRepositoryItem) -> None:
        pass


class ProductRepositoryItem(ParameterGroup):
    def __init__(
        self,
        parent: ProductRepositoryItemObserver,
        metadata_item: MetadataRepositoryItem,
        scan_item: ScanRepositoryItem,
        geometry: ProductGeometry,
        probe_item: ProbeRepositoryItem,
        object_item: ObjectRepositoryItem,
        validator: ProductValidator,
        losses: Sequence[LossValue],
    ) -> None:
        super().__init__()
        self._parent = parent
        self._metadata_item = metadata_item
        self._scan_item = scan_item
        self._geometry = geometry
        self._probe_item = probe_item
        self._object_item = object_item
        self._validator = validator
        self._losses = list(losses)

        self._add_group('metadata', self._metadata_item, observe=True)
        self._add_group('scan', self._scan_item, observe=True)
        self._add_group('probe', self._probe_item, observe=True)
        self._add_group('object', self._object_item, observe=True)

        self._index = -1  # used by ProductRepository

    def assign(self, product: Product) -> None:
        self._metadata_item.assign(product.metadata)
        self._scan_item.assign(product.positions)
        self._probe_item.assign(product.probes)
        self._object_item.assign(product.object_)
        self._losses = list(product.losses)
        self._parent.handle_losses_changed(self)

    def sync_to_settings(self) -> None:
        self._metadata_item.sync_to_settings()
        self._scan_item.sync_to_settings()
        self._probe_item.sync_to_settings()
        self._object_item.sync_to_settings()

    def get_name(self) -> str:
        return self._metadata_item.name.get_value()

    def set_name(self, name: str) -> None:
        self._metadata_item.name.set_value(name)

    def get_metadata_item(self) -> MetadataRepositoryItem:
        return self._metadata_item

    def get_scan_item(self) -> ScanRepositoryItem:
        return self._scan_item

    def get_geometry(self) -> ProductGeometry:
        return self._geometry

    def get_probe_item(self) -> ProbeRepositoryItem:
        return self._probe_item

    def get_object_item(self) -> ObjectRepositoryItem:
        return self._object_item

    def _invalidate_losses(self) -> None:
        self._losses = list()
        self._parent.handle_losses_changed(self)

    def get_losses(self) -> Sequence[LossValue]:
        return self._losses

    def get_product(self) -> Product:
        return Product(
            metadata=self._metadata_item.get_metadata(),
            positions=self._scan_item.get_scan(),
            probes=self._probe_item.get_probes(),
            object_=self._object_item.get_object(),
            losses=self.get_losses(),
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata_item:
            self._invalidate_losses()
            self._parent.handle_metadata_changed(self)
        elif observable is self._scan_item:
            self._invalidate_losses()
            self._parent.handle_scan_changed(self)
        elif observable is self._probe_item:
            self._invalidate_losses()
            self._parent.handle_probe_changed(self)
        elif observable is self._object_item:
            self._invalidate_losses()
            self._parent.handle_object_changed(self)
        else:
            super()._update(observable)


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
    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        pass

    @abstractmethod
    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        pass
