from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging
import sys

from ...api.observer import ObservableSequence
from ...api.plugins import PluginChooser
from ..metadata import MetadataRepositoryItem
from ..object import ObjectBuilderFactory, ObjectRepositoryItem
from ..probe import ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .repository import (ExperimentRepository, ExperimentRepositoryItem,
                         ExperimentRepositoryObserver)

logger = logging.getLogger(__name__)


class ObjectRepository(ObservableSequence[ObjectRepositoryItem], ExperimentRepositoryObserver):

    def __init__(self, repository: ExperimentRepository, factory: ObjectBuilderFactory) -> None:
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

    def __getitem__(self, index: int | slice) -> ObjectRepositoryItem | Sequence[ObjectRepositoryItem]:
        if isinstance(index, slice):
            return [item.getObject() for item in self._repository[index]]
        else:
            return self._repository[index].getObject()

    def __len__(self) -> int:
        return len(self._repository)

    def openObject(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._factory.createObjectFromFile(filePath, fileFilter)

        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to open object {index}!')
        else:
            item.setBuilder(builder)

    def saveObject(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self[index]
        except IndexError:
            logger.warning(f'Failed to save object {index}!')
        else:
            self._factory.saveObject(filePath, fileFilter, item.getObject())

    def handleItemInserted(self, index: int, item: ExperimentRepositoryItem) -> None:
        self.notifyObserversItemInserted(index, item.getObject())

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        self.notifyObserversItemChanged(index, item)

    def handleItemRemoved(self, index: int, item: ExperimentRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getObject())
