from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import threading
import time

from ptychodus.api.fluorescence import (
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceFileWriter,
)
from ptychodus.api.observer import Observable
from ptychodus.api.product import Product

from ..task_monitor import NotifyObserversTask, TaskProgressMonitor
from ..task_manager import ForegroundTaskManager

__all__ = [
    'EnhanceFluorescenceBackgroundTask',
    'FluorescenceDatasetEmitter',
    'FluorescenceTaskMonitor',
]

logger = logging.getLogger(__name__)


class FluorescenceDatasetEmitter(Observable):
    def __init__(self, foreground_task_manager: ForegroundTaskManager) -> None:
        super().__init__()
        self._foreground_task_manager = foreground_task_manager
        self._lock = threading.Lock()
        self._latest_enhanced: FluorescenceDataset | None = None

    def get_latest_enhanced(self) -> FluorescenceDataset | None:
        with self._lock:
            return self._latest_enhanced

    def _publish(self, dataset: FluorescenceDataset) -> None:
        with self._lock:
            self._latest_enhanced = dataset

        self._foreground_task_manager.put_foreground_task(NotifyObserversTask(self))


class FluorescenceTaskMonitor(TaskProgressMonitor):
    def __init__(self, foreground_task_manager: ForegroundTaskManager) -> None:
        super().__init__(foreground_task_manager)
        self._dataset_emitter = FluorescenceDatasetEmitter(foreground_task_manager)

    def get_dataset_emitter(self) -> FluorescenceDatasetEmitter:
        return self._dataset_emitter

    def update_enhanced(self, dataset: FluorescenceDataset) -> None:
        self._dataset_emitter._publish(dataset)


@dataclass(frozen=True)
class EnhanceFluorescenceBackgroundTask:
    task_monitor: FluorescenceTaskMonitor
    enhancer: FluorescenceEnhancer
    dataset: FluorescenceDataset
    product: Product
    output_file_path: Path | None
    output_file_writer: FluorescenceFileWriter | None

    def __call__(self) -> None:
        with self.task_monitor as monitor:
            progress_goal = self.enhancer.get_progress_goal()
            monitor.update_progress(0, progress_goal)
            tic = time.perf_counter()
            last_dataset: FluorescenceDataset | None = None
            parameters = FluorescenceEnhancerInput(dataset=self.dataset, product=self.product)

            for output in self.enhancer.enhance(parameters):
                monitor.update_progress(output.progress, progress_goal)
                monitor.update_enhanced(output.dataset)
                last_dataset = output.dataset

                if monitor.is_stopping:
                    break

            toc = time.perf_counter()
            logger.info(f'Enhancement time {toc - tic:.4f} seconds.')

            if (
                last_dataset is not None
                and not monitor.is_stopping
                and self.output_file_path is not None
                and self.output_file_writer is not None
            ):
                logger.debug(f'Writing enhanced fluorescence to "{self.output_file_path}"')
                self.output_file_writer.write(self.output_file_path, last_dataset)
