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
    def __init__(
        self,
        foreground_task_manager: ForegroundTaskManager,
        processing_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        super().__init__()
        self._foreground_task_manager = foreground_task_manager
        self._processing_event = processing_event
        self._stop_event = stop_event

        self._log_handler = ProcessingLogHandler()
        self._log_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        )
        self._progress_lock = threading.Lock()
        self._progress_goal = 0
        self._progress = 0

    def get_log_handler(self) -> ProcessingLogHandler:
        return self._log_handler

    @property
    def is_processing(self) -> bool:
        return self._processing_event.is_set()

    def stop_processing(self) -> None:
        self._stop_event.set()

    def get_progress(self) -> int:
        with self._progress_lock:
            return self._progress

    def get_progress_goal(self) -> int:
        with self._progress_lock:
            return self._progress_goal

    def _notify_observers_foreground(self) -> None:
        task = NotifyObserversTask(self)
        self._foreground_task_manager.put_foreground_task(task)

    def _update_progress(self, progress: int, progress_goal: int) -> None:
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


class ProcessingContext:
    def __init__(self, foreground_task_manager: ForegroundTaskManager) -> None:
        self._foreground_task_manager = foreground_task_manager
        self._processing_event = threading.Event()
        self._stop_event = threading.Event()
        self._progress_monitor = ProcessingProgressMonitor(
            foreground_task_manager, self._processing_event, self._stop_event
        )

    def get_progress_monitor(self) -> ProcessingProgressMonitor:
        return self._progress_monitor

    def get_log_handler(self) -> ProcessingLogHandler:
        return self._progress_monitor.get_log_handler()

    def update_progress(self, progress: int, progress_goal: int) -> None:
        self._progress_monitor._update_progress(progress, progress_goal)

    def update_product(
        self, product_item: ProductRepositoryItem, result: ReconstructOutput
    ) -> None:
        # TODO log handler messages to product comments
        task = UpdateProductTask(product_item, result.product)
        self._foreground_task_manager.put_foreground_task(task)

    @property
    def is_stopping(self) -> bool:
        return self._stop_event.is_set()

    def __enter__(self) -> ProcessingContext:
        self._processing_event.set()
        self._stop_event.clear()
        self._progress_monitor._notify_observers_foreground()
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
        self._processing_event.clear()
        self._progress_monitor._notify_observers_foreground()


@dataclass(frozen=True)
class ReconstructBackgroundTask:
    context: ProcessingContext
    reconstructor: Reconstructor
    reconstruct_input: ReconstructInput
    product_item: ProductRepositoryItem
    output_product_file: Path | None
    finished_event: threading.Event

    def __call__(self) -> None:
        with self.context as context:
            context.update_progress(0, self.reconstructor.get_progress_goal())
            tic = time.perf_counter()

            for result in self.reconstructor.reconstruct(self.reconstruct_input):
                context.update_progress(result.progress, self.reconstructor.get_progress_goal())
                context.update_product(self.product_item, result)
                of = self.output_product_file

                if of is not None:
                    of = of.parent / f'{of.stem}.{result.progress:06d}{of.suffix}'
                    save_product(of, result.product)

                if context.is_stopping:
                    break

            toc = time.perf_counter()
            logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')

        self.finished_event.set()


@dataclass(frozen=True)
class TrainBackgroundTask:
    context: ProcessingContext
    trainer: TrainableReconstructor
    reconstruct_input: ReconstructInput
    input_path: Path
    output_path: Path
    finished_event: threading.Event

    def __call__(self) -> None:
        with self.context as context:
            context.update_progress(0, self.trainer.get_progress_goal())
            tic = time.perf_counter()

            for result in self.trainer.train(self.input_path, self.output_path):
                context.update_progress(result.progress, self.trainer.get_progress_goal())

                if context.is_stopping:
                    break

            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds.')

        self.finished_event.set()
