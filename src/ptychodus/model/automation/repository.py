from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import overload
import logging

from ptychodus.api.observer import ObservableSequence

from .settings import AutomationSettings

logger = logging.getLogger(__name__)


class AutomationDatasetState(Enum):
    FOUND = auto()
    WAITING = auto()
    RUNNING = auto()
    COMPLETE = auto()


@dataclass(frozen=True)
class AutomationDataset:
    file_path: Path
    state: AutomationDatasetState
    label: str


class AutomationDatasetRepository(ObservableSequence[AutomationDataset]):
    def __init__(self, settings: AutomationSettings) -> None:
        super().__init__()
        self._settings = settings
        self._dataset_list: list[AutomationDataset] = list()

    @overload
    def __getitem__(self, index: int) -> AutomationDataset: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[AutomationDataset]: ...

    def __getitem__(self, index: int | slice) -> AutomationDataset | Sequence[AutomationDataset]:
        return self._dataset_list[index]

    def __len__(self) -> int:
        return len(self._dataset_list)

    def upsert(self, file_path: Path, state: AutomationDatasetState) -> None:
        try:
            index = next(
                idx for idx, item in enumerate(self._dataset_list) if item.file_path == file_path
            )
        except StopIteration:
            index = len(self._dataset_list)
            label = str(file_path.relative_to(self._settings.data_directory.get_value()))
            dataset = AutomationDataset(file_path, state, label)
            self._dataset_list.append(dataset)
            self.notify_observers_item_inserted(index, dataset)
        else:
            old_dataset = self._dataset_list[index]
            new_dataset = AutomationDataset(file_path, state, old_dataset.label)
            logger.debug(f'{file_path}: {old_dataset.state} -> {new_dataset.state}')
            self._dataset_list[index] = new_dataset
            self.notify_observers_item_changed(index, new_dataset)

    def clear(self) -> None:
        while True:
            try:
                dataset = self._dataset_list.pop()
            except IndexError:
                break
            else:
                self.notify_observers_item_removed(len(self._dataset_list), dataset)


class UpdateAutomationDatasetRepository:
    def __init__(
        self,
        repository: AutomationDatasetRepository,
        file_path: Path,
        state: AutomationDatasetState,
    ) -> None:
        self._repository = repository
        self._file_path = file_path
        self._state = state

    def __call__(self) -> None:
        self._repository.upsert(self._file_path, self._state)
