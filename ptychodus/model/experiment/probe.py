from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging

from ...api.observer import ObservableSequence
from ..metadata import MetadataRepositoryItem
from ..object import ObjectRepositoryItem
from ..probe import ProbeBuilderFactory, ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .repository import (ExperimentRepository, ExperimentRepositoryItem,
                         ExperimentRepositoryObserver)

logger = logging.getLogger(__name__)


class ProbeRepository(ObservableSequence[ProbeRepositoryItem], ExperimentRepositoryObserver):

    def __init__(self, repository: ExperimentRepository, factory: ProbeBuilderFactory) -> None:
        super().__init__()
        self._repository = repository
        self._repository.addObserver(self)
        self._factory = factory

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

    def openProbe(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._factory.createProbeFromFile(filePath, fileFilter)

        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to open probe {index}!')
        else:
            item.setBuilder(builder)

    def saveProbe(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to save probe {index}!')
        else:
            self._factory.saveProbe(filePath, fileFilter, item.getProbe())

    def handleItemInserted(self, index: int, item: ExperimentRepositoryItem) -> None:
        self.notifyObserversItemInserted(index, item.getProbe())

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        self.notifyObserversItemChanged(index, item)

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ExperimentRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getProbe())
