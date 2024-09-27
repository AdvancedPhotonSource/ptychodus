from pathlib import Path
from time import monotonic as time
import logging
import queue
import threading

from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

from .repository import AutomationDatasetRepository, AutomationDatasetState
from .settings import AutomationSettings

logger = logging.getLogger(__name__)


class AutomationDatasetProcessor:
    def __init__(
        self,
        settings: AutomationSettings,
        repository: AutomationDatasetRepository,
        workflow: FileBasedWorkflow,
        workflowAPI: WorkflowAPI,
        processingQueue: queue.Queue[Path],
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._workflow = workflow
        self._workflowAPI = workflowAPI
        self._processingQueue = processingQueue
        self._stopWorkEvent = threading.Event()
        self._worker = threading.Thread()
        self._nextJobTime = time()

    @property
    def isAlive(self) -> bool:
        return self._worker.is_alive()

    def put(self, filePath: Path) -> None:
        self._repository.put(filePath, AutomationDatasetState.WAITING)
        self._processingQueue.put(filePath)

    def runOnce(self) -> None:
        try:
            filePath = self._processingQueue.get(block=False)

            try:
                self._repository.put(filePath, AutomationDatasetState.PROCESSING)
                self._workflow.execute(self._workflowAPI, filePath)
                self._repository.put(filePath, AutomationDatasetState.COMPLETE)
            except Exception:
                logger.exception('Error while processing dataset!')
            finally:
                self._processingQueue.task_done()
        except queue.Empty:
            pass

    def _run(self) -> None:
        while not self._stopWorkEvent.is_set():
            try:
                filePath = self._processingQueue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            delayInSeconds = self._nextJobTime - time()

            if delayInSeconds > 0.0 and self._stopWorkEvent.wait(timeout=delayInSeconds):
                break

            try:
                self._repository.put(filePath, AutomationDatasetState.PROCESSING)
                self._workflow.execute(self._workflowAPI, filePath)
                self._repository.put(filePath, AutomationDatasetState.COMPLETE)
            except Exception:
                logger.exception('Error while processing dataset!')
            finally:
                self._processingQueue.task_done()
                self._nextJobTime = self._settings.processingIntervalInSeconds.getValue() + time()

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
