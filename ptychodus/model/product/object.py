from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging

from ...api.observer import ObservableSequence
from ...api.visualize import Plot2D
from ..metadata import MetadataRepositoryItem
from ..object import ObjectBuilderFactory, ObjectRepositoryItem
from ..probe import ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .repository import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver

logger = logging.getLogger(__name__)


class ObjectRepository(ObservableSequence[ObjectRepositoryItem], ProductRepositoryObserver):

    def __init__(self, repository: ProductRepository, factory: ObjectBuilderFactory) -> None:
        super().__init__()
        self._repository = repository
        self._repository.addObserver(self)
        self._factory = factory

    @overload
    def __getitem__(self, index: int) -> ObjectRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ObjectRepositoryItem]:
        ...

    def __getitem__(self,
                    index: int | slice) -> ObjectRepositoryItem | Sequence[ObjectRepositoryItem]:
        if isinstance(index, slice):
            return [item.getObject() for item in self._repository[index]]
        else:
            return self._repository[index].getObject()

    def __len__(self) -> int:
        return len(self._repository)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._factory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._factory.getOpenFileFilter()

    def openObject(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._factory.createObjectFromFile(filePath, fileFilter)

        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to open object {index}!')
        else:
            item.setBuilder(builder)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._factory.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._factory.getSaveFileFilter()

    def saveObject(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to save object {index}!')
        else:
            self._factory.saveObject(filePath, fileFilter, item.getObject())

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemInserted(index, item.getObject())

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        self.notifyObserversItemChanged(index, item)

    def handleCostsChanged(self, index: int, costs: Plot2D) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getObject())
