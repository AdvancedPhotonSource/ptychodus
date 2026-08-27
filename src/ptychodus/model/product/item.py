from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parameters import ParameterGroup
from ptychodus.api.product import LossValue, Product

from ..diffraction import AssembledDiffractionDataset, DiffractionDatasetObserver
from .geometry import ProductGeometry
from .metadata import MetadataRepositoryItem, UniqueNameFactory
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .probe_positions import ProbePositionsRepositoryItem

logger = logging.getLogger(__name__)


class ProductState(Enum):
    READY = 'ready'
    PENDING = 'pending'
    FAILED = 'failed'


class ProductRepositoryItemObserver(UniqueNameFactory):
    @abstractmethod
    def handle_metadata_changed(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_probe_positions_changed(self, item: ProductRepositoryItem) -> None:
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

    @abstractmethod
    def handle_dataset_changed(self, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_state_changed(self, item: ProductRepositoryItem) -> None:
        pass


class ProductRepositoryItem(ParameterGroup):
    def __init__(
        self,
        parent: ProductRepositoryItemObserver,
        metadata_item: MetadataRepositoryItem,
        probe_positions_item: ProbePositionsRepositoryItem,
        geometry: ProductGeometry,
        probe_item: ProbeRepositoryItem,
        object_item: ObjectRepositoryItem,
        losses: Sequence[LossValue],
        dataset: AssembledDiffractionDataset | None = None,
        state: ProductState = ProductState.READY,
    ) -> None:
        super().__init__()
        self._parent = parent
        self._metadata_item = metadata_item
        self._probe_positions_item = probe_positions_item
        self._geometry = geometry
        self._probe_item = probe_item
        self._object_item = object_item
        self._losses = list(losses)
        self._dataset: AssembledDiffractionDataset | None = None
        self._dataset_observer: _BoundDatasetObserver | None = None
        self._state: ProductState = state

        self._add_group('metadata', self._metadata_item, observe=True)
        self._add_group('probe_positions', self._probe_positions_item, observe=True)
        self._add_group('probe', self._probe_item, observe=True)
        self._add_group('object', self._object_item, observe=True)

        self._index = -1  # used by ProductRepository

        # Bind the geometry's detector extent + pixel geometry to the initial
        # dataset (if any) so downstream probe/object sizes are correct from the
        # start.
        if dataset is not None:
            self._bind_dataset(dataset)

    def assign(self, product: Product) -> None:
        self._metadata_item.assign(product.metadata)
        self._probe_positions_item.assign(product.probe_positions)
        self._probe_item.assign(product.probes)
        self._object_item.assign(product.object_)
        self._losses = list(product.losses)
        self._parent.handle_losses_changed(self)

    def copy_contents_from(
        self,
        source: ProductRepositoryItem,
    ) -> None:
        """Copy inner state from a freshly-built source item into this stub.

        Uses each subgroup's assign_item so the stub's subgroup identities are
        preserved (peripheral scan/probe/object repositories continue to observe
        the same subgroup instances they registered at insert time). The item's
        index in the ProductRepository never changes.
        """
        self._metadata_item.assign(source._metadata_item.get_metadata())
        # Bind the dataset (and thus the detector extent on the geometry) BEFORE
        # rebuilding probe/object subgroups — their _rebuild() otherwise sees an
        # invalid pixel geometry and silently no-ops, leaving them empty.
        # _insert_via_queue only routes here when source has a bound dataset.
        assert source._dataset is not None
        self.bind_dataset(source._dataset)
        self._probe_positions_item.assign_item(source._probe_positions_item)
        self._probe_item.assign_item(source._probe_item)
        self._object_item.assign_item(source._object_item)
        self._losses = list(source._losses)
        self._parent.handle_losses_changed(self)

    def sync_to_settings(self) -> None:
        self._metadata_item.sync_to_settings()
        self._probe_positions_item.sync_to_settings()
        self._probe_item.sync_to_settings()
        self._object_item.sync_to_settings()

    def get_name(self) -> str:
        return self._metadata_item.name.get_value()

    def set_name(self, name: str) -> None:
        self._metadata_item.name.set_value(name)

    def get_metadata_item(self) -> MetadataRepositoryItem:
        return self._metadata_item

    def get_probe_positions_item(self) -> ProbePositionsRepositoryItem:
        return self._probe_positions_item

    def get_geometry(self) -> ProductGeometry:
        return self._geometry

    def get_probe_item(self) -> ProbeRepositoryItem:
        return self._probe_item

    def get_object_item(self) -> ObjectRepositoryItem:
        return self._object_item

    def get_dataset(self) -> AssembledDiffractionDataset | None:
        """Return the diffraction dataset this product is associated with, if any.

        Model-only reference; not part of the persisted Product.
        """
        return self._dataset

    def bind_dataset(self, dataset: AssembledDiffractionDataset) -> None:
        if self._dataset is not dataset:
            self._bind_dataset(dataset)
            self._parent.handle_dataset_changed(self)

    def unbind_dataset(self) -> None:
        if self._dataset is not None:
            self._bind_dataset(None)
            self._parent.handle_dataset_changed(self)

    def _bind_dataset(self, dataset: AssembledDiffractionDataset | None) -> None:
        # Detach the previous dataset's observer before rebinding.
        if self._dataset is not None and self._dataset_observer is not None:
            self._dataset.remove_observer(self._dataset_observer)
            self._dataset_observer = None

        self._dataset = dataset

        if dataset is None:
            self._geometry.set_detector_extent(None)
            self._geometry.set_detector_pixel_geometry(None)
            return

        self._geometry.set_detector_extent(dataset.get_metadata().detector_extent)
        self._geometry.set_detector_pixel_geometry(dataset.get_raw_pixel_geometry())

        # Mirror future edits (pixel geometry, reload) from the dataset back into
        # the geometry so probe/object sizes stay in sync.
        self._dataset_observer = _BoundDatasetObserver(self)
        dataset.add_observer(self._dataset_observer)

        self._auto_estimate_probe_photon_count()

    def _auto_estimate_probe_photon_count(self) -> None:
        # Guard on the default sentinel so a user- or file-supplied value is
        # never overwritten. A real measurement never sums to exactly zero on
        # the brightest pattern.
        if self._dataset is None:
            return
        if self._metadata_item.probe_photon_count.get_value() != 0.0:
            return
        photon_count = self._dataset.get_assembled_data().get_probe_photon_count()
        self._metadata_item.probe_photon_count.set_value(float(photon_count))

    def _sync_geometry_from_dataset(self) -> None:
        if self._dataset is None:
            return
        self._geometry.set_detector_extent(self._dataset.get_metadata().detector_extent)
        self._geometry.set_detector_pixel_geometry(self._dataset.get_raw_pixel_geometry())

    def get_state(self) -> ProductState:
        return self._state

    def is_pending(self) -> bool:
        return self._state is ProductState.PENDING

    def is_failed(self) -> bool:
        return self._state is ProductState.FAILED

    def set_state(self, state: ProductState) -> None:
        if self._state != state:
            self._state = state
            self._parent.handle_state_changed(self)

    def _invalidate_losses(self) -> None:
        self._losses = list()
        self._parent.handle_losses_changed(self)

    def get_losses(self) -> Sequence[LossValue]:
        return self._losses

    def get_product(self) -> Product:
        return Product(
            metadata=self._metadata_item.get_metadata(),
            probe_positions=self._probe_positions_item.get_probe_positions(),
            probes=self._probe_item.get_probes(),
            object_=self._object_item.get_object(),
            losses=self.get_losses(),
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata_item:
            self._invalidate_losses()
            self._parent.handle_metadata_changed(self)
        elif observable is self._probe_positions_item:
            self._invalidate_losses()
            self._parent.handle_probe_positions_changed(self)
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
    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
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
    def handle_dataset_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_state_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        pass


class _BoundDatasetObserver(DiffractionDatasetObserver):
    """Mirrors dataset changes back into the bound product's geometry."""

    def __init__(self, item: ProductRepositoryItem) -> None:
        super().__init__()
        self._item = item

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self._item._sync_geometry_from_dataset()
        self._item._auto_estimate_probe_photon_count()

    def handle_pixel_geometry_changed(self) -> None:
        self._item._sync_geometry_from_dataset()
