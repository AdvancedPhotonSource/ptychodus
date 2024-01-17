from collections.abc import Sequence
from typing import overload
import logging
import sys

from ...api.observer import ObservableSequence
from ..metadata import MetadataRepositoryItem
from ..object import ObjectRepositoryItem
from ..probe import ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .repository import (ExperimentRepository, ExperimentRepositoryItem,
                         ExperimentRepositoryObserver)

logger = logging.getLogger(__name__)


class MetadataRepository(ObservableSequence[MetadataRepositoryItem], ExperimentRepositoryObserver):

    def __init__(self, repository: ExperimentRepository) -> None:
        super().__init__()
        self._repository = repository
        self._repository.addObserver(self)

    @overload
    def __getitem__(self, index: int) -> MetadataRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[MetadataRepositoryItem]:
        ...

    def __getitem__(
            self, index: int | slice) -> MetadataRepositoryItem | Sequence[MetadataRepositoryItem]:
        if isinstance(index, slice):
            return [item.getMetadata() for item in self._repository[index]]
        else:
            return self._repository[index].getMetadata()

    def __len__(self) -> int:
        return len(self._repository)

    def handleItemInserted(self, index: int, item: ExperimentRepositoryItem) -> None:
        self.notifyObserversItemInserted(index, item.getMetadata())

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        self.notifyObserversItemChanged(index, item)

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ExperimentRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getMetadata())

    def getInfoText(self) -> str:
        sizeInMB = sum(sys.getsizeof(exp) for exp in self._repository) / (1024 * 1024)
        return f'Total: {len(self)} [{sizeInMB:.2f}MB]'
