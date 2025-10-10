from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import overload
import logging
import threading
import time

from ptychodus.api.observer import Observable
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput, Reconstructor

from ..product import ProductRepositoryItem
from ..task_manager import ForegroundTaskManager
from .log import ReconstructorLogHandler

__all__ = [
    'ReconstructorProgressMonitor',
    'ReconstructorContext',
    'ReconstructBackgroundTask',
    'TrainBackgroundTask',
]

logger = logging.getLogger(__name__)


class UpdateProductTask:
    def __init__(self, product_item: ProductRepositoryItem, product: Product) -> None:
        self._product_item = product_item
        self._product = product

    def __call__(self) -> None:
        name = self._product_item.get_name()
        self._product_item.assign(self._product)
        self._product_item.set_name(name)


class ReconstructorProgressMonitor(Observable):
    def __init__(self, log_handler: ReconstructorLogHandler) -> None:
        super().__init__()
        self._log_handler = log_handler
        self._is_reconstructing = False
        self._progress_goal = 0
        self._progress = 0
        self._lock = threading.Lock()
        self._changed = threading.Event()

    def _set_reconstructing(self, is_reconstructing: bool) -> None:
        self._is_reconstructing = is_reconstructing
        self.notify_observers()

    @property
    def is_reconstructing(self) -> bool:
        return self._is_reconstructing

    def message_log(self) -> Iterator[str]:
        return self._log_handler.messages()

    def set_progress_goal(self, progress_goal: int) -> None:
        with self._lock:
            if self._progress_goal != progress_goal:
                self._progress_goal = progress_goal
                self._changed.set()

    def get_progress_goal(self) -> int:
        with self._lock:
            return self._progress_goal

    def set_progress(self, progress: int) -> None:
        with self._lock:
            if self._progress != progress:
                self._progress = progress
                self._changed.set()

    def get_progress(self) -> int:
        with self._lock:
            return self._progress

    def notify_observers_if_changed(self) -> None:
        # only call this method from the main thread
        if self._changed.is_set():
            self._changed.clear()
            self.notify_observers()


class ReconstructorContext:
    def __init__(
        self, log_handler: ReconstructorLogHandler, foreground_task_manager: ForegroundTaskManager
    ) -> None:
        self._foreground_task_manager = foreground_task_manager
        self._progress_monitor = ReconstructorProgressMonitor(log_handler)
        self._is_idle = threading.Event()
        self._is_idle.set()

    def wait_for_reconstruction(self, timeout_sec: float | None = None) -> None:
        self._is_idle.wait(timeout_sec)

    def get_progress_monitor(self) -> ReconstructorProgressMonitor:
        return self._progress_monitor

    def update_progress(
        self, product_item: ProductRepositoryItem, result: ReconstructOutput
    ) -> None:
        task = UpdateProductTask(product_item, result.product)
        self._foreground_task_manager.put_foreground_task(task)
        self._progress_monitor.set_progress(result.progress)

    def notify_observers_if_progress_changed(self) -> None:
        self._progress_monitor.notify_observers_if_changed()

    def __enter__(self) -> ReconstructorContext:
        self._is_idle.clear()
        self._progress_monitor._set_reconstructing(True)
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
        self._progress_monitor._set_reconstructing(False)
        self._is_idle.set()


@dataclass(frozen=True)
class ReconstructBackgroundTask:
    context: ReconstructorContext
    reconstructor: Reconstructor
    parameters: ReconstructInput
    product_item: ProductRepositoryItem

    def __call__(self) -> None:
        with self.context as context:
            progress_monitor = context.get_progress_monitor()
            progress_monitor.set_progress_goal(self.reconstructor.get_progress_goal())
            tic = time.perf_counter()

            for result in self.reconstructor.reconstruct(self.parameters):
                context.update_progress(self.product_item, result)

            toc = time.perf_counter()
            logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')


class TrainBackgroundTask:  # TODO
    def __call__(self) -> None:
        pass
