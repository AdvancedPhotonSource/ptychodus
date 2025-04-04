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
        workflow_api: WorkflowAPI,
        processing_queue: queue.Queue[Path],
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._workflow = workflow
        self._workflow_api = workflow_api
        self._processing_queue = processing_queue
        self._stop_work_event = threading.Event()
        self._worker = threading.Thread()
        self._next_job_time = time()

    @property
    def is_alive(self) -> bool:
        return self._worker.is_alive()

    def put(self, file_path: Path) -> None:
        self._repository.put(file_path, AutomationDatasetState.WAITING)
        self._processing_queue.put(file_path)

    def run_once(self) -> None:
        try:
            file_path = self._processing_queue.get(block=False)

            try:
                self._repository.put(file_path, AutomationDatasetState.PROCESSING)
                self._workflow.execute(self._workflow_api, file_path)
                self._repository.put(file_path, AutomationDatasetState.COMPLETE)
            except Exception:
                logger.exception('Error while processing dataset!')
            finally:
                self._processing_queue.task_done()
        except queue.Empty:
            pass

    def _run(self) -> None:
        while not self._stop_work_event.is_set():
            try:
                file_path = self._processing_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            delay_s = self._next_job_time - time()

            if delay_s > 0.0 and self._stop_work_event.wait(timeout=delay_s):
                break

            try:
                self._repository.put(file_path, AutomationDatasetState.PROCESSING)
                self._workflow.execute(self._workflow_api, file_path)
                self._repository.put(file_path, AutomationDatasetState.COMPLETE)
            except Exception:
                logger.exception('Error while processing dataset!')
            finally:
                self._processing_queue.task_done()
                self._next_job_time = self._settings.processing_interval_s.get_value() + time()

    def start(self) -> None:
        self.stop()
        logger.info('Starting automation processor thread...')
        self._stop_work_event.clear()
        self._worker = threading.Thread(target=self._run)
        self._worker.start()
        logger.info('Automation processor thread started.')

    def stop(self) -> None:
        if self.is_alive:
            logger.info('Stopping automation processor thread...')
            self._stop_work_event.set()
            self._worker.join()
            logger.info('Automation processor thread stopped.')
