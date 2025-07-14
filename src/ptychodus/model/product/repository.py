from collections.abc import Sequence
from typing import overload
import logging
import sys

from ptychodus.api.units import BYTES_PER_MEGABYTE

from .item import ProductRepositoryItem, ProductRepositoryItemObserver, ProductRepositoryObserver

logger = logging.getLogger(__name__)


class ProductRepository(Sequence[ProductRepositoryItem], ProductRepositoryItemObserver):
    def __init__(self) -> None:
        super().__init__()
        self._item_list: list[ProductRepositoryItem] = []
        self._observer_list: list[ProductRepositoryObserver] = []

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

    def create_unique_name(self, candidate_name: str) -> str:
        reserved_names = set([item.get_name() for item in self._item_list])
        name = candidate_name or 'Unnamed'
        match = 0

        while name in reserved_names:
            match += 1
            name = f'{candidate_name}-{match}'

        return name

    def _update_indexes(self) -> None:
        for index, item in enumerate(self._item_list):
            item._index = index

    def insert_product(self, item: ProductRepositoryItem) -> int:
        index = len(self._item_list)
        self._item_list.append(item)

        self._update_indexes()

        for observer in self._observer_list:
            observer.handle_item_inserted(index, item)

        return index

    def remove_product(self, index: int) -> None:
        try:
            item = self._item_list.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove product item {index}!')
            return

        self._update_indexes()

        for observer in self._observer_list:
            observer.handle_item_removed(index, item)

    def get_info_text(self) -> str:
        size_MB = sum(sys.getsizeof(prod) for prod in self._item_list) / BYTES_PER_MEGABYTE  # noqa: N806
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
        index = item._index

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_metadata_changed(index, metadata)

    def handle_scan_changed(self, item: ProductRepositoryItem) -> None:
        index = item._index
        scan = item.get_scan_item()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_scan_changed(index, scan)

    def handle_probe_changed(self, item: ProductRepositoryItem) -> None:
        index = item._index
        probe = item.get_probe_item()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_probe_changed(index, probe)

    def handle_object_changed(self, item: ProductRepositoryItem) -> None:
        index = item._index
        object_ = item.get_object_item()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_object_changed(index, object_)

    def handle_losses_changed(self, item: ProductRepositoryItem) -> None:
        index = item._index
        losses = item.get_losses()

        if index < 0:
            logger.warning(f'Failed to look up index for "{item.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_losses_changed(index, losses)
