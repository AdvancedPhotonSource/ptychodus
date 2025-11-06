from abc import ABC, abstractmethod
from typing import Callable, Final, TypeAlias
import logging
import queue
import threading

logger = logging.getLogger(__name__)

ForegroundTask: TypeAlias = Callable[[], None]


class ForegroundTaskManager(ABC):
    @abstractmethod
    def put_foreground_task(self, task: ForegroundTask) -> None:
        pass

    @property
    @abstractmethod
    def foreground_queue_size(self) -> int:
        pass


BackgroundTask: TypeAlias = Callable[[], ForegroundTask | None]


class BackgroundTaskManager(ABC):
    @abstractmethod
    def put_background_task(self, task: BackgroundTask) -> None:
        pass

    @property
    @abstractmethod
    def background_queue_size(self) -> int:
        pass


class TaskManager(BackgroundTaskManager, ForegroundTaskManager):
    WAIT_TIME_S: Final[float] = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._background_queue: queue.Queue[BackgroundTask] = queue.Queue()
        self._foreground_queue: queue.Queue[ForegroundTask] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    @property
    def is_stopping(self) -> bool:
        return self._stop_event.is_set()

    def put_background_task(self, task: BackgroundTask) -> None:
        self._background_queue.put(task)

    @property
    def background_queue_size(self) -> int:
        return self._background_queue.qsize()

    def _run_background_tasks(self) -> None:
        while not self._stop_event.is_set():
            try:
                background_task = self._background_queue.get(block=True, timeout=self.WAIT_TIME_S)
            except queue.Empty:
                continue

            try:
                foreground_task = background_task()
            except Exception:
                logger.exception(f'Background task exception during {background_task}!')
            else:
                if foreground_task is not None:
                    self._foreground_queue.put(foreground_task)
            finally:
                self._background_queue.task_done()

    def put_foreground_task(self, task: ForegroundTask) -> None:
        self._foreground_queue.put(task)

    @property
    def foreground_queue_size(self) -> int:
        return self._foreground_queue.qsize()

    def run_foreground_tasks(self) -> None:
        while True:
            try:
                task = self._foreground_queue.get(block=False)
            except queue.Empty:
                break

            try:
                task()
            except Exception:
                logger.exception(f'Foreground task exception during {task}!')
            finally:
                self._foreground_queue.task_done()

    def start(self) -> None:
        if self._worker is None:
            logger.info('Starting task manager...')
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._run_background_tasks)
            self._worker.start()
            logger.info('Task manager started.')
        else:
            logger.warning('Worker already started!')

    def _stop(self) -> None:
        if self._worker is None:
            logger.warning('Worker is None!')
        else:
            logger.info('Stopping task manager...')
            self._stop_event.set()
            self._worker.join()
            self._worker = None
            logger.info('Task manager stopped.')

    def stop(self, *, await_finish: bool) -> None:
        if self._stop_event.is_set():
            logger.info('Task manager already stopped.')
        else:
            if await_finish:
                logger.info('Finishing tasks...')
                self._background_queue.join()
                logger.info('Tasks finished.')

            self._stop()
