from collections.abc import Sequence
from typing import overload
import logging

from ptychodus.api.observer import ObservableSequence

from .item import ProductRepositoryItem, ProductRepositoryObserver
from .metadata import MetadataRepositoryItem
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .repository import ProductRepository
from .scan import ScanRepositoryItem

logger = logging.getLogger(__name__)


class ProbeRepository(ObservableSequence[ProbeRepositoryItem], ProductRepositoryObserver):
    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository
        self._repository.add_observer(self)

    def get_name(self, index: int) -> str:
        return self._repository[index].get_name()

    def set_name(self, index: int, name: str) -> None:
        self._repository[index].set_name(name)

    @overload
    def __getitem__(self, index: int) -> ProbeRepositoryItem: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProbeRepositoryItem]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ProbeRepositoryItem | Sequence[ProbeRepositoryItem]:
        if isinstance(index, slice):
            return [item.get_probe_item() for item in self._repository[index]]
        else:
            return self._repository[index].get_probe_item()

    def __len__(self) -> int:
        return len(self._repository)

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        self.notify_observers_item_inserted(index, item.get_probe_item())

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handle_scan_changed(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        self.notify_observers_item_changed(index, item)

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_costs_changed(self, index: int, costs: Sequence[float]) -> None:
        pass

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        self.notify_observers_item_removed(index, item.get_probe_item())
