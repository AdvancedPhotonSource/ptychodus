from collections.abc import Sequence
from typing import overload
import logging

from ptychodus.api.observer import ObservableSequence

from .item import ProductRepositoryItem, ProductRepositoryObserver
from .metadata import MetadataRepositoryItem
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .productRepository import ProductRepository
from .scan import ScanRepositoryItem

logger = logging.getLogger(__name__)


class ScanRepository(ObservableSequence[ScanRepositoryItem], ProductRepositoryObserver):
    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository
        self._repository.addObserver(self)

    def getName(self, index: int) -> str:
        return self._repository[index].getName()

    def setName(self, index: int, name: str) -> None:
        self._repository[index].setName(name)

    @overload
    def __getitem__(self, index: int) -> ScanRepositoryItem: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ScanRepositoryItem]: ...

    def __getitem__(self, index: int | slice) -> ScanRepositoryItem | Sequence[ScanRepositoryItem]:
        if isinstance(index, slice):
            return [item.getScan() for item in self._repository[index]]
        else:
            return self._repository[index].getScan()

    def __len__(self) -> int:
        return len(self._repository)

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemInserted(index, item.getScan())

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        self.notifyObserversItemChanged(index, item)

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handleCostsChanged(self, index: int, costs: Sequence[float]) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getScan())
