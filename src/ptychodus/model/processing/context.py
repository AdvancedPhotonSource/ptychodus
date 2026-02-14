from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import overload
import logging
import queue
import threading
import time

from ptychodus.api.io import save_product
from ptychodus.api.observer import Observable
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
    TrainableReconstructor,
)

from ..product import ProductRepositoryItem
from ..task_manager import ForegroundTaskManager

__all__ = [
    'ProcessingProgressMonitor',
    'ProcessingContext',
    'ReconstructBackgroundTask',
    'TrainBackgroundTask',
]

logger = logging.getLogger(__name__)


class NotifyObserversTask:
    def __init__(self, observable: Observable) -> None:
        self._observable = observable

    def __call__(self) -> None:
        self._observable.notify_observers()


class UpdateProductTask:
    def __init__(self, product_item: ProductRepositoryItem, product: Product) -> None:
        self._product_item = product_item
        self._product = product

    def __call__(self) -> None:
        name = self._product_item.get_name()
        self._product_item.assign(self._product)
        self._product_item.set_name(name)


class ProcessingLogHandler(logging.Handler):
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


class ProcessingProgressMonitor(Observable):
    def __init__(self, foreground_task_manager: ForegroundTaskManager) -> None:
        super().__init__()
        self._foreground_task_manager = foreground_task_manager
        self._log_handler = ProcessingLogHandler()
        self._log_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )
        self._is_processing = False
        self._progress_goal = 0
        self._progress = 0
        self._lock = threading.Lock()
        self._changed = threading.Event()

    def get_log_handler(self) -> logging.Handler:
        return self._log_handler

    def _notify_observers_foreground(self) -> None:
        task = NotifyObserversTask(self)
        self._foreground_task_manager.put_foreground_task(task)

    def _set_processing(self, is_processing: bool) -> None:
        with self._lock:
            self._is_processing = is_processing
            self._notify_observers_foreground()

    @property
    def is_processing(self) -> bool:
        with self._lock:
            return self._is_processing

    def messages(self) -> Iterator[str]:
        return self._log_handler.messages()

    def set_progress_goal(self, progress_goal: int) -> None:
        with self._lock:
            if self._progress_goal != progress_goal:
                self._progress_goal = progress_goal
                self._notify_observers_foreground()

    def get_progress_goal(self) -> int:
        with self._lock:
            return self._progress_goal

    def set_progress(self, progress: int) -> None:
        with self._lock:
            if self._progress != progress:
                self._progress = progress
                self._notify_observers_foreground()

    def get_progress(self) -> int:
        with self._lock:
            return self._progress


class ProcessingContext:
    def __init__(self, foreground_task_manager: ForegroundTaskManager) -> None:
        self._foreground_task_manager = foreground_task_manager
        self._progress_monitor = ProcessingProgressMonitor(foreground_task_manager)

    def get_progress_monitor(self) -> ProcessingProgressMonitor:
        return self._progress_monitor

    def get_log_handler(self) -> logging.Handler:
        return self._progress_monitor.get_log_handler()

    def update_progress(
        self, product_item: ProductRepositoryItem, result: ReconstructOutput
    ) -> None:
        task = UpdateProductTask(product_item, result.product)
        self._foreground_task_manager.put_foreground_task(task)
        self._progress_monitor.set_progress(result.progress)

    def __enter__(self) -> ProcessingContext:
        self._progress_monitor._set_processing(True)
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
        self._progress_monitor._set_processing(False)


@dataclass(frozen=True)
class ReconstructBackgroundTask:
    context: ProcessingContext
    reconstructor: Reconstructor
    parameters: ReconstructInput
    product_item: ProductRepositoryItem
    finished_event: threading.Event
    output_product_file: Path | None

    def __call__(self) -> None:
        with self.context as context:
            progress_monitor = context.get_progress_monitor()
            progress_monitor.set_progress_goal(self.reconstructor.get_progress_goal())
            tic = time.perf_counter()

            for result in self.reconstructor.reconstruct(self.parameters):
                context.update_progress(self.product_item, result)
                of = self.output_product_file

                if of is not None:
                    of = of.parent / f'{of.stem}.{result.progress:06d}.{of.suffix}'
                    save_product(of, result.product)

            toc = time.perf_counter()
            logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')

        self.finished_event.set()


@dataclass(frozen=True)
class TrainBackgroundTask:
    context: ProcessingContext
    trainer: TrainableReconstructor
    reconstruct_input: ReconstructInput
    finished_event: threading.Event
    input_path: Path
    output_path: Path

    def __call__(self) -> None:
        with self.context as context:
            progress_monitor = context.get_progress_monitor()
            progress_monitor.set_progress_goal(self.trainer.get_progress_goal())
            tic = time.perf_counter()

            for result in self.trainer.train(self.input_path, self.output_path):
                progress_monitor.set_progress(len(result.training_loss))

            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds.')

        self.finished_event.set()
