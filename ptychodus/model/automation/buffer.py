from collections import OrderedDict
from pathlib import Path
from time import monotonic as time
import logging
import threading

from .repository import AutomationDatasetRepository, AutomationDatasetState
from .settings import AutomationSettings

logger = logging.getLogger(__name__)


class AutomationDatasetBuffer:

    def __init__(self, settings: AutomationSettings,
                 repository: AutomationDatasetRepository) -> None:
        self._settings = settings
        self._repository = repository
        self._eventTimes: OrderedDict[Path, float] = OrderedDict()
        self._eventTimesLock = threading.Lock()
        self._stopWorkEvent = threading.Event()
        self._worker = threading.Thread()

    def put(self, filePath: Path) -> None:
        with self._eventTimesLock:
            self._eventTimes[filePath] = time()
            self._eventTimes.move_to_end(filePath)

        self._repository.put(filePath, AutomationDatasetState.CREATED)

    def _process(self) -> None:
        while not self._stopWorkEvent.is_set():
            isFileReadyForProcessing = False

            with self._eventTimesLock:
                try:
                    filePath, eventTime = next(iter(self._eventTimes.items()))
                except StopIteration:
                    pass
                else:
                    delayTime = self._settings.watchdogDelayInSeconds.value
                    isFileReadyForProcessing = (eventTime + delayTime < time())

            if isFileReadyForProcessing:
                self._repository.put(filePath, AutomationDatasetState.WAITING)
            else:
                self._stopWorkEvent.wait(timeout=5.)  # TODO make configurable

    def start(self) -> None:
        # TODO call me
        if self._worker.is_alive():
            self.stop()

        logger.info('Starting data assembler...')
        self._stopWorkEvent.clear()
        self._worker = threading.Thread(target=self._process)
        self._worker.start()
        logger.info('Data assembler started.')

    def stop(self) -> None:
        # TODO call me
        logger.info('Stopping automation thread...')
        self._stopWorkEvent.set()
        self._worker.join()
        logger.info('Automation thread stopped.')
