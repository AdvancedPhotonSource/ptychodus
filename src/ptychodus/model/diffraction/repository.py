from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import overload
import logging

from ptychodus.api.constants import format_bytes

from ..task_manager import TaskManager
from .dataset import AssembledDiffractionDataset
from .monitor import DiffractionTaskMonitor
from .settings import DetectorSettings, DiffractionSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)


def build_default_factory(
    diffraction_settings: DiffractionSettings,
    pattern_sizer: PatternSizer,
    detector_settings: DetectorSettings,
    task_manager: TaskManager,
    task_monitor: DiffractionTaskMonitor,
) -> Callable[[str], AssembledDiffractionDataset]:
    def _factory(name: str) -> AssembledDiffractionDataset:
        return AssembledDiffractionDataset(
            diffraction_settings,
            pattern_sizer,
            detector_settings,
            task_manager,
            task_monitor,
            name=name,
        )

    return _factory


class DiffractionDatasetRepositoryObserver(ABC):
    @abstractmethod
    def handle_dataset_inserted(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        pass

    @abstractmethod
    def handle_dataset_removed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        pass

    @abstractmethod
    def handle_metadata_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        pass

    @abstractmethod
    def handle_state_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        pass


class DiffractionDatasetRepository(Sequence[AssembledDiffractionDataset]):
    def __init__(
        self,
        factory: Callable[[str], AssembledDiffractionDataset] | None = None,
    ) -> None:
        super().__init__()
        self._factory = factory
        self._dataset_list: list[AssembledDiffractionDataset] = []
        self._observer_list: list[DiffractionDatasetRepositoryObserver] = []

    def create_dataset(self, name: str) -> AssembledDiffractionDataset:
        if self._factory is None:
            raise RuntimeError(
                'DiffractionDatasetRepository was constructed without a factory; '
                'cannot build new datasets.'
            )
        unique_name = self.create_unique_name(name)
        return self._factory(unique_name)

    @overload
    def __getitem__(self, index: int) -> AssembledDiffractionDataset: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[AssembledDiffractionDataset]: ...

    def __getitem__(
        self, index: int | slice
    ) -> AssembledDiffractionDataset | Sequence[AssembledDiffractionDataset]:
        return self._dataset_list[index]

    def __len__(self) -> int:
        return len(self._dataset_list)

    def add_observer(self, observer: DiffractionDatasetRepositoryObserver) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: DiffractionDatasetRepositoryObserver) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def create_unique_name(self, candidate_name: str) -> str:
        reserved_names = {dataset.get_name() for dataset in self._dataset_list}
        name = candidate_name or 'Unnamed'
        match = 0

        while name in reserved_names:
            match += 1
            name = f'{candidate_name}-{match}'

        return name

    def insert_dataset(self, dataset: AssembledDiffractionDataset) -> int:
        index = len(self._dataset_list)
        self._dataset_list.append(dataset)

        for observer in self._observer_list:
            observer.handle_dataset_inserted(index, dataset)

        return index

    def remove_dataset(self, index: int) -> None:
        try:
            dataset = self._dataset_list.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove dataset {index}!')
            return

        dataset.clear()

        for observer in self._observer_list:
            observer.handle_dataset_removed(index, dataset)

    def clear(self) -> None:
        for idx in range(len(self._dataset_list) - 1, -1, -1):
            self.remove_dataset(idx)

    def handle_metadata_changed(self, dataset: AssembledDiffractionDataset) -> None:
        index = self._dataset_row(dataset)

        if index is None:
            logger.warning(f'Failed to look up index for "{dataset.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_metadata_changed(index, dataset)

    def handle_state_changed(self, dataset: AssembledDiffractionDataset) -> None:
        index = self._dataset_row(dataset)

        if index is None:
            logger.warning(f'Failed to look up index for "{dataset.get_name()}"!')
            return

        for observer in self._observer_list:
            observer.handle_state_changed(index, dataset)

    def _dataset_row(self, dataset: AssembledDiffractionDataset) -> int | None:
        try:
            return self._dataset_list.index(dataset)
        except ValueError:
            return None

    def get_info_text(self) -> str:
        nbytes = sum(dataset.get_nbytes() for dataset in self._dataset_list)
        return f'Datasets: {len(self)} [{format_bytes(nbytes)}]'
