from pathlib import Path
import logging
import queue
import threading

from .repository import AutomationDatasetRepository, AutomationDatasetState
from .settings import AutomationSettings
from .workflow import AutomationDatasetWorkflow

logger = logging.getLogger(__name__)


class AutomationDatasetProcessor:

    def __init__(self, settings: AutomationSettings, repository: AutomationDatasetRepository,
                 workflow: AutomationDatasetWorkflow, processingQueue: queue.Queue[Path]) -> None:
        self._settings = settings
        self._repository = repository
        self._workflow = workflow
        self._processingQueue = processingQueue
        self._stopWorkEvent = threading.Event()
        self._worker = threading.Thread()

    @property
    def isAlive(self) -> bool:
        return self._worker.is_alive()

    def _run(self) -> None:
        while not self._stopWorkEvent.is_set():
            try:
                filePath = self._processingQueue.get(block=True, timeout=1)

                try:
                    self._repository.put(filePath, AutomationDatasetState.PROCESSING)
                    self._workflow.execute(filePath)
                    self._repository.put(filePath, AutomationDatasetState.COMPLETE)
                finally:
                    self._processingQueue.task_done()
            except queue.Empty:
                pass
            except:
                logger.exception('Error while processing dataset!')

    def start(self) -> None:
        self.stop()
        logger.info('Starting automation processor thread...')
        self._stopWorkEvent.clear()
        self._worker = threading.Thread(target=self._run)
        self._worker.start()
        logger.info('Automation processor thread started.')

    def stop(self) -> None:
        if self.isAlive:
            logger.info('Stopping automation processor thread...')
            self._stopWorkEvent.set()
            self._worker.join()
            logger.info('Automation processor thread stopped.')
