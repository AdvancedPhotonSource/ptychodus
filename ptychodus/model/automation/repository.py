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
        self._fileList: list[Path] = list()
        self._fileState: dict[Path, AutomationDatasetState] = dict()
        self._lock = threading.Lock()
        self._changedEvent = threading.Event()

    def put(self, filePath: Path, state: AutomationDatasetState) -> None:
        with self._lock:
            try:
                priorState = self._fileState[filePath]
            except KeyError:
                if state == AutomationDatasetState.EXISTS:
                    self._fileList.append(filePath)
                    self._fileState[filePath] = state
                else:
                    logger.error(f'{filePath}: UNKNOWN -> {state}')
            else:
                logger.debug(f'{filePath}: {priorState} -> {state}')
                self._fileState[filePath] = state

            self._changedEvent.set()

    def clear(self) -> None:
        with self._lock:
            self._fileList.clear()
            self._fileState.clear()

            self._changedEvent.set()

    def getLabel(self, index: int) -> str:
        with self._lock:
            filePath = self._fileList[index]
            return str(filePath.relative_to(self._settings.dataDirectory.getValue()))

    def getState(self, index: int) -> AutomationDatasetState:
        with self._lock:
            filePath = self._fileList[index]
            return self._fileState[filePath]

    def __len__(self) -> int:
        with self._lock:
            return len(self._fileList)

    def notifyObserversIfRepositoryChanged(self) -> None:
        if self._changedEvent.is_set():
            self._changedEvent.clear()
            self.notifyObservers()
