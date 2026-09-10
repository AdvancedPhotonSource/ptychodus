from collections.abc import Sequence
from typing import overload
import logging

from ptychodus.api.constants import format_bytes
from ptychodus.api.observer import ObservableSequence
from ptychodus.api.product import LossValue

from ..diffraction import AssembledDiffractionDataset
from .item import ProductRepositoryItem, ProductRepositoryObserver
from .metadata import MetadataRepositoryItem
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .repository import ProductRepository
from .probe_positions import ProbePositionsRepositoryItem

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

    def get_dataset(self, index: int) -> AssembledDiffractionDataset | None:
        return self._repository[index].get_dataset()

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

    def get_info_text(self) -> str:
        nbytes = sum(item.get_probes().nbytes for item in self)
        return f'Probes: {len(self)} [{format_bytes(nbytes)}]'

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        self.notify_observers_item_inserted(index, item.get_probe_item())

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        self.notify_observers_item_changed(index, item)

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        pass

    def handle_dataset_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_state_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        self.notify_observers_item_removed(index, item.get_probe_item())
