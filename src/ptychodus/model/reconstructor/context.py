from __future__ import annotations
from dataclasses import dataclass
from types import TracebackType
from typing import overload
import logging
import threading
import time

from ptychodus.api.product import Product
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput, Reconstructor

from ..product import ProductRepositoryItem
from ..task_manager import ForegroundTaskManager

__all__ = ['ReconstructorContext', 'ReconstructBackgroundTask', 'TrainBackgroundTask']

logger = logging.getLogger(__name__)


class UpdateProductTask:
    def __init__(self, product_item: ProductRepositoryItem, product: Product) -> None:
        self._product_item = product_item
        self._product = product

    def __call__(self) -> None:
        name = self._product_item.get_name()
        self._product_item.assign(self._product)
        self._product_item.set_name(name)


class ReconstructorContext:
    def __init__(
        self,
        foreground_task_manager: ForegroundTaskManager,
    ) -> None:
        self._foreground_task_manager = foreground_task_manager
        self._progress_goal = 0
        self._progress = 0
        self._lock = threading.Lock()
        self._is_idle = threading.Event()
        self._is_idle.set()

    @property
    def is_reconstructing(self) -> bool:
        return not self._is_idle.is_set()

    def wait_for_reconstruction(self, timeout_sec: float | None = None) -> None:
        self._is_idle.wait(timeout_sec)

    def set_progress_goal(self, progress_goal: int) -> None:
        with self._lock:
            self._progress_goal = progress_goal

    def get_progress_goal(self) -> int:
        with self._lock:
            return self._progress_goal

    def get_progress(self) -> int:
        with self._lock:
            return self._progress

    def update_progress(
        self, product_item: ProductRepositoryItem, result: ReconstructOutput
    ) -> None:
        with self._lock:
            self._progress = result.progress

        task = UpdateProductTask(product_item, result.product)
        self._foreground_task_manager.put_foreground_task(task)

    def __enter__(self) -> ReconstructorContext:
        self._is_idle.clear()
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
        self._is_idle.set()


@dataclass(frozen=True)
class ReconstructBackgroundTask:
    context: ReconstructorContext
    reconstructor: Reconstructor
    parameters: ReconstructInput
    product_item: ProductRepositoryItem

    def __call__(self) -> None:
        with self.context as context:
            context.set_progress_goal(self.reconstructor.get_progress_goal())
            tic = time.perf_counter()

            for result in self.reconstructor.reconstruct(self.parameters):
                context.update_progress(self.product_item, result)

            toc = time.perf_counter()
            logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')


class TrainBackgroundTask:  # TODO
    def __call__(self) -> None:
        pass
