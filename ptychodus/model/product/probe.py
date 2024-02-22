from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import overload
import logging

from ...api.observer import ObservableSequence
from ...api.visualize import Plot2D
from ..metadata import MetadataRepositoryItem
from ..object import ObjectRepositoryItem
from ..probe import ProbeBuilderFactory, ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .repository import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver

logger = logging.getLogger(__name__)


class ProbeRepository(ObservableSequence[ProbeRepositoryItem], ProductRepositoryObserver):

    def __init__(self, repository: ProductRepository, factory: ProbeBuilderFactory) -> None:
        super().__init__()
        self._repository = repository
        self._repository.addObserver(self)
        self._factory = factory

    def getName(self, index: int) -> str:
        return self._repository[index].getName()

    def setName(self, index: int, name: str) -> None:
        self._repository[index].setName(name)

    @overload
    def __getitem__(self, index: int) -> ProbeRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProbeRepositoryItem]:
        ...

    def __getitem__(self,
                    index: int | slice) -> ProbeRepositoryItem | Sequence[ProbeRepositoryItem]:
        if isinstance(index, slice):
            return [item.getProbe() for item in self._repository[index]]
        else:
            return self._repository[index].getProbe()

    def __len__(self) -> int:
        return len(self._repository)

    def builderNames(self) -> Iterator[str]:
        return iter(self._factory)

    def setBuilderByName(self, index: int, builderName: str) -> bool:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return False

        try:
            builder = self._factory.create(builderName, item.getGeometry())
        except KeyError:
            logger.warning(f'Failed to create builder {builderName}!')
            return False

        item.getProbe().setBuilder(builder)
        return True

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._factory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._factory.getOpenFileFilter()

    def openProbe(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._factory.createProbeFromFile(filePath, fileFilter)

        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to open probe {index}!')
        else:
            item.setBuilder(builder)

    def copyProbe(self, sourceIndex: int, destinationIndex: int) -> None:
        print(f'Copy {sourceIndex} -> {destinationIndex}')  # FIXME

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._factory.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._factory.getSaveFileFilter()

    def saveProbe(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to save probe {index}!')
        else:
            self._factory.saveProbe(filePath, fileFilter, item.getProbe())

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemInserted(index, item.getProbe())

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        self.notifyObserversItemChanged(index, item)

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handleCostsChanged(self, index: int, costs: Plot2D) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getProbe())
