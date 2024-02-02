from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging

from ...api.observer import ObservableSequence
from ...api.visualize import Plot2D
from ..metadata import MetadataRepositoryItem
from ..object import ObjectRepositoryItem
from ..probe import ProbeRepositoryItem
from ..scan import ScanBuilderFactory, ScanRepositoryItem
from .repository import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver

logger = logging.getLogger(__name__)


class ScanRepository(ObservableSequence[ScanRepositoryItem], ProductRepositoryObserver):

    def __init__(self, repository: ProductRepository, factory: ScanBuilderFactory) -> None:
        super().__init__()
        self._repository = repository
        self._repository.addObserver(self)
        self._factory = factory

    @overload
    def __getitem__(self, index: int) -> ScanRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ScanRepositoryItem]:
        ...

    def __getitem__(self, index: int | slice) -> ScanRepositoryItem | Sequence[ScanRepositoryItem]:
        if isinstance(index, slice):
            return [item.getScan() for item in self._repository[index]]
        else:
            return self._repository[index].getScan()

    def __len__(self) -> int:
        return len(self._repository)

    def openScan(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._factory.createScanFromFile(filePath, fileFilter)

        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to open scan {index}!')
        else:
            item.setBuilder(builder)

    def saveScan(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to save scan {index}!')
        else:
            self._factory.saveScan(filePath, fileFilter, item.getScan())

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

    def handleCostsChanged(self, index: int, costs: Plot2D) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getScan())
