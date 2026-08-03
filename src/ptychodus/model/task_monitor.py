from __future__ import annotations
from collections.abc import Iterator
from types import TracebackType
from typing import Self, overload
import logging
import queue
import threading

from ptychodus.api.observer import Observable

from .task_manager import ForegroundTaskManager

__all__ = [
    'NotifyObserversTask',
    'TaskLogHandler',
    'TaskProgressMonitor',
]


class NotifyObserversTask:
    def __init__(self, observable: Observable) -> None:
        self._observable = observable

    def __call__(self) -> None:
        self._observable.notify_observers()


class TaskLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._log: queue.Queue[str] = queue.Queue()

    def messages(self) -> Iterator[str]:
        while True:
            try:
                yield self._log.get(block=False)
                self._log.task_done()
            except queue.Empty:
                break

    def emit(self, record: logging.LogRecord) -> None:
        text = self.format(record)
        self._log.put(text)


class TaskProgressMonitor(Observable):
    def __init__(self, foreground_task_manager: ForegroundTaskManager) -> None:
        super().__init__()
        self._foreground_task_manager = foreground_task_manager

        self._log_handler = TaskLogHandler()
        self._log_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )

        self._state_condition = threading.Condition()
        self._is_processing = False
        self._is_stopping = False
        self._completed_runs = 0
        self._last_error: BaseException | None = None

        self._progress_lock = threading.Lock()
        self._progress_goal = 0
        self._progress = 0

    def get_log_handler(self) -> TaskLogHandler:
        return self._log_handler

    @property
    def is_processing(self) -> bool:
        with self._state_condition:
            return self._is_processing

    @property
    def is_stopping(self) -> bool:
        with self._state_condition:
            return self._is_stopping

    def stop_processing(self) -> None:
        with self._state_condition:
            self._is_stopping = True

    def get_progress(self) -> int:
        with self._progress_lock:
            return self._progress

    def get_progress_goal(self) -> int:
        with self._progress_lock:
            return self._progress_goal

    def update_progress(self, progress: int, progress_goal: int) -> None:
        with self._progress_lock:
            is_changed = False

            if self._progress != progress:
                self._progress = progress
                is_changed = True

            if self._progress_goal != progress_goal:
                self._progress_goal = progress_goal
                is_changed = True

            if is_changed:
                self._notify_observers_foreground()

    def get_completed_runs(self) -> int:
        """Snapshot the number of completed task runs. Use with wait_for_completion_after."""
        with self._state_condition:
            return self._completed_runs

    def wait_for_completion_after(self, snapshot: int, timeout: float | None = None) -> bool:
        """Block until a task run has completed after the given snapshot. Returns True if one did."""
        with self._state_condition:
            return self._state_condition.wait_for(
                lambda: self._completed_runs > snapshot, timeout=timeout
            )

    def get_last_error(self) -> BaseException | None:
        """Return the exception captured by the most recent task run, if any."""
        with self._state_condition:
            return self._last_error

    def raise_if_failed(self) -> None:
        """Re-raise the exception captured by the most recent task run, if any."""
        with self._state_condition:
            error = self._last_error
        if error is not None:
            raise error

    def _notify_observers_foreground(self) -> None:
        task = NotifyObserversTask(self)
        self._foreground_task_manager.put_foreground_task(task)

    def __enter__(self) -> Self:
        with self._state_condition:
            self._is_processing = True
            self._is_stopping = False
            self._last_error = None
            self._state_condition.notify_all()
        self._notify_observers_foreground()
        return self

    @overload
    def __exit__(self, exception_type: None, exception_value: None, traceback: None) -> None: ...

    @overload
    def __exit__(
        self,
        exception_type: type[BaseException],
        exception_value: BaseException,
        traceback: TracebackType,
    ) -> None: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        with self._state_condition:
            self._is_processing = False
            self._completed_runs += 1
            if exception_value is not None:
                self._last_error = exception_value
            self._state_condition.notify_all()
        self._notify_observers_foreground()
