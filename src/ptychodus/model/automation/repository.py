from enum import Enum, auto
from pathlib import Path
import logging
import threading

from ptychodus.api.observer import Observable

from .settings import AutomationSettings

logger = logging.getLogger(__name__)


class AutomationDatasetState(Enum):
    EXISTS = auto()
    WAITING = auto()
    PROCESSING = auto()
    COMPLETE = auto()


class AutomationDatasetRepository(Observable):
    def __init__(self, settings: AutomationSettings) -> None:
        super().__init__()
        self._settings = settings
        self._file_list: list[Path] = list()
        self._file_state: dict[Path, AutomationDatasetState] = dict()
        self._lock = threading.Lock()
        self._changed_event = threading.Event()

    def put(self, file_path: Path, state: AutomationDatasetState) -> None:
        with self._lock:
            try:
                prior_state = self._file_state[file_path]
            except KeyError:
                if state == AutomationDatasetState.EXISTS:
                    self._file_list.append(file_path)
                    self._file_state[file_path] = state
                else:
                    logger.error(f'{file_path}: UNKNOWN -> {state}')
            else:
                logger.debug(f'{file_path}: {prior_state} -> {state}')
                self._file_state[file_path] = state

            self._changed_event.set()

    def clear(self) -> None:
        with self._lock:
            self._file_list.clear()
            self._file_state.clear()

            self._changed_event.set()

    def get_label(self, index: int) -> str:
        with self._lock:
            file_path = self._file_list[index]
            return str(file_path.relative_to(self._settings.data_directory.get_value()))

    def get_state(self, index: int) -> AutomationDatasetState:
        with self._lock:
            file_path = self._file_list[index]
            return self._file_state[file_path]

    def __len__(self) -> int:
        with self._lock:
            return len(self._file_list)

    def notify_observers_if_repository_changed(self) -> None:
        if self._changed_event.is_set():
            self._changed_event.clear()
            self.notify_observers()
