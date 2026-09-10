from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import overload
import logging

from ptychodus.api.constants import format_bytes
from ptychodus.api.typing import RealArrayType
from ptychodus.api.fluorescence import FluorescenceDataset

from ..product import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver
from ..product.metadata import MetadataRepositoryItem
from ..product.object import ObjectRepositoryItem
from ..product.probe import ProbeRepositoryItem
from ..product.probe_positions import ProbePositionsRepositoryItem
from ptychodus.api.product import LossValue

logger = logging.getLogger(__name__)


class FluorescenceItemState(Enum):
    READY = 'ready'
    ENHANCING = 'enhancing'
    FAILED = 'failed'
    ORPHANED = 'orphaned'


class FluorescenceRepositoryItemObserver(ABC):
    @abstractmethod
    def handle_metadata_changed(self, item: FluorescenceRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_enhanced_changed(self, item: FluorescenceRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_state_changed(self, item: FluorescenceRepositoryItem) -> None:
        pass


class FluorescenceRepositoryItem:
    def __init__(
        self,
        parent: FluorescenceRepositoryItemObserver,
        *,
        name: str,
        product: ProductRepositoryItem,
        measured: FluorescenceDataset,
        source_path: Path | None = None,
        source_file_type: str | None = None,
    ) -> None:
        self._parent = parent
        self._name = name
        self._product = product
        self._measured = measured
        self._source_path = source_path
        self._source_file_type = source_file_type
        self._enhanced: FluorescenceDataset | None = None
        self._measured_summary_cache: RealArrayType | None = None
        self._enhanced_summary_cache: RealArrayType | None = None
        self._state = FluorescenceItemState.READY
        self._index = -1  # used by FluorescenceRepository

    def get_name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        if self._name != name:
            self._name = name
            self._parent.handle_metadata_changed(self)

    def get_product(self) -> ProductRepositoryItem:
        return self._product

    def set_product(self, product: ProductRepositoryItem) -> None:
        """Re-bind this item to a different product.

        An ORPHANED item returns to READY: the state means "the product this pointed at
        was removed", and re-binding answers that. A FAILED item keeps its state, which
        records an enhancement failure rather than a missing product.
        """
        if self._product is product:
            return

        self._product = product

        if self._state is FluorescenceItemState.ORPHANED:
            self._state = FluorescenceItemState.READY
            self._parent.handle_state_changed(self)

        self._parent.handle_metadata_changed(self)

    def get_source_path(self) -> Path | None:
        return self._source_path

    def get_source_file_type(self) -> str | None:
        return self._source_file_type

    def get_measured(self) -> FluorescenceDataset:
        return self._measured

    def get_enhanced(self) -> FluorescenceDataset | None:
        return self._enhanced

    def get_nbytes(self) -> int:
        """Bytes held by the measured and enhanced element maps.

        The derived summary caches are excluded so that element-map sizes sum exactly
        to their item, and items sum exactly to the repository total.
        """
        sz = self._measured.nbytes

        if self._enhanced is not None:
            sz += self._enhanced.nbytes

        return sz

    def set_enhanced(self, dataset: FluorescenceDataset) -> None:
        self._enhanced = dataset
        # Enhanced was just (re)set — drop the cached summary so the next read
        # recomputes from the fresh element maps.
        self._enhanced_summary_cache = None
        self._parent.handle_enhanced_changed(self)

    @staticmethod
    def _sum_element_maps(dataset: FluorescenceDataset) -> RealArrayType | None:
        if not len(dataset):
            return None
        total = dataset[0].counts_per_second.copy()
        for element_map in dataset[1:]:
            total += element_map.counts_per_second
        return total

    def get_measured_summary(self) -> RealArrayType | None:
        """Elementwise sum across the measured element maps (cached, immutable)."""
        if self._measured_summary_cache is None:
            self._measured_summary_cache = self._sum_element_maps(self._measured)
        return self._measured_summary_cache

    def get_enhanced_summary(self) -> RealArrayType | None:
        """Elementwise sum across the enhanced element maps, or None if not enhanced."""
        if self._enhanced is None:
            return None
        if self._enhanced_summary_cache is None:
            self._enhanced_summary_cache = self._sum_element_maps(self._enhanced)
        return self._enhanced_summary_cache

    def get_state(self) -> FluorescenceItemState:
        return self._state

    def set_state(self, state: FluorescenceItemState) -> None:
        if self._state is not state:
            self._state = state
            self._parent.handle_state_changed(self)

    def mark_orphaned(self) -> None:
        # Idempotent. Re-adding the same product does not clear this by itself — the
        # stored reference is to the old instance — but set_product() does.
        if self._state is not FluorescenceItemState.ORPHANED:
            self._state = FluorescenceItemState.ORPHANED
            self._parent.handle_state_changed(self)


class FluorescenceRepositoryObserver(ABC):
    @abstractmethod
    def handle_item_inserted(self, index: int, item: FluorescenceRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_item_removed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_metadata_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_enhanced_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        pass

    @abstractmethod
    def handle_state_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        pass


class _ProductRemovalAdapter(ProductRepositoryObserver):
    """Bridges ProductRepositoryObserver → FluorescenceRepository._on_product_removed.

    The only signal we care about is product removal; all other callbacks are
    no-ops. Kept separate so FluorescenceRepository doesn't have to implement
    every method of the ProductRepositoryObserver ABC just to catch removals.
    """

    def __init__(self, repository: FluorescenceRepository) -> None:
        super().__init__()
        self._repository = repository

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        pass

    def handle_dataset_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_state_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        self._repository._on_product_removed(item)


class FluorescenceRepository(
    Sequence[FluorescenceRepositoryItem], FluorescenceRepositoryItemObserver
):
    def __init__(self, product_repository: ProductRepository) -> None:
        super().__init__()
        self._product_repository = product_repository
        self._item_list: list[FluorescenceRepositoryItem] = []
        self._observer_list: list[FluorescenceRepositoryObserver] = []
        # Adapter kept as an instance attribute so the observer registration
        # survives for the lifetime of the repository.
        self._product_removal_adapter = _ProductRemovalAdapter(self)
        product_repository.add_observer(self._product_removal_adapter)

    @overload
    def __getitem__(self, index: int) -> FluorescenceRepositoryItem: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[FluorescenceRepositoryItem]: ...

    def __getitem__(
        self, index: int | slice
    ) -> FluorescenceRepositoryItem | Sequence[FluorescenceRepositoryItem]:
        return self._item_list[index]

    def __len__(self) -> int:
        return len(self._item_list)

    def create_unique_name(self, candidate_name: str) -> str:
        reserved_names = {item.get_name() for item in self._item_list}
        name = candidate_name or 'Unnamed'
        match = 0

        while name in reserved_names:
            match += 1
            name = f'{candidate_name}-{match}'

        return name

    def _update_indexes(self) -> None:
        for index, item in enumerate(self._item_list):
            item._index = index

    def insert_item(self, item: FluorescenceRepositoryItem) -> int:
        index = len(self._item_list)
        self._item_list.append(item)
        self._update_indexes()

        for observer in self._observer_list:
            observer.handle_item_inserted(index, item)

        return index

    def remove_item(self, index: int) -> None:
        try:
            item = self._item_list.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove fluorescence item {index}!')
            return

        self._update_indexes()

        for observer in self._observer_list:
            observer.handle_item_removed(index, item)

    def _on_product_removed(self, removed_product: ProductRepositoryItem) -> None:
        """Mark every fluorescence item bound to the removed product as orphaned.

        Items retain their strong reference to the (now-detached) product, so
        rendering and saving keep working; only re-enhancement is blocked.
        """
        for item in self._item_list:
            if item.get_product() is removed_product:
                item.mark_orphaned()

    def get_info_text(self) -> str:
        nbytes = sum(item.get_nbytes() for item in self._item_list)
        return f'Datasets: {len(self)} [{format_bytes(nbytes)}]'

    def add_observer(self, observer: FluorescenceRepositoryObserver) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: FluorescenceRepositoryObserver) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def handle_metadata_changed(self, item: FluorescenceRepositoryItem) -> None:
        index = item._index
        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return
        for observer in self._observer_list:
            observer.handle_metadata_changed(index, item)

    def handle_enhanced_changed(self, item: FluorescenceRepositoryItem) -> None:
        index = item._index
        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return
        for observer in self._observer_list:
            observer.handle_enhanced_changed(index, item)

    def handle_state_changed(self, item: FluorescenceRepositoryItem) -> None:
        index = item._index
        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return
        for observer in self._observer_list:
            observer.handle_state_changed(index, item)
