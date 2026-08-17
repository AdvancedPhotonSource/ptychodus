from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from ptychodus.api.fluorescence import (
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceFileWriter,
)
from ptychodus.api.product import Product

from ..task_monitor import TaskProgressMonitor
from .repository import FluorescenceItemState, FluorescenceRepositoryItem

__all__ = [
    'EnhanceFluorescenceBackgroundTask',
    'FluorescenceTaskMonitor',
]

logger = logging.getLogger(__name__)


class UpdateEnhancedTask:
    """Foreground-scheduled application of a new enhanced dataset onto its item."""

    def __init__(self, item: FluorescenceRepositoryItem, dataset: FluorescenceDataset) -> None:
        self._item = item
        self._dataset = dataset

    def __call__(self) -> None:
        self._item.set_enhanced(self._dataset)


class SetItemStateTask:
    """Foreground-scheduled state transition on a FluorescenceRepositoryItem.

    Skips the transition if the item is already ORPHANED — the product was
    removed mid-enhancement and the orphan flag must win over the task's own
    READY/FAILED verdict.
    """

    def __init__(
        self, item: FluorescenceRepositoryItem, target_state: FluorescenceItemState
    ) -> None:
        self._item = item
        self._target_state = target_state

    def __call__(self) -> None:
        if self._item.get_state() is FluorescenceItemState.ORPHANED:
            return
        self._item.set_state(self._target_state)


class FluorescenceTaskMonitor(TaskProgressMonitor):
    def update_enhanced(
        self, item: FluorescenceRepositoryItem, dataset: FluorescenceDataset
    ) -> None:
        task = UpdateEnhancedTask(item, dataset)
        self._foreground_task_manager.put_foreground_task(task)

    def transition_item_state(
        self, item: FluorescenceRepositoryItem, target_state: FluorescenceItemState
    ) -> None:
        task = SetItemStateTask(item, target_state)
        self._foreground_task_manager.put_foreground_task(task)


@dataclass(frozen=True)
class EnhanceFluorescenceBackgroundTask:
    task_monitor: FluorescenceTaskMonitor
    enhancer: FluorescenceEnhancer
    dataset: FluorescenceDataset
    product: Product
    item: FluorescenceRepositoryItem
    output_file_path: Path | None
    output_file_writer: FluorescenceFileWriter | None

    def __call__(self) -> None:
        try:
            with self.task_monitor as monitor:
                progress_goal = self.enhancer.get_progress_goal()
                monitor.update_progress(0, progress_goal)
                tic = time.perf_counter()
                last_dataset: FluorescenceDataset | None = None
                parameters = FluorescenceEnhancerInput(dataset=self.dataset, product=self.product)

                for output in self.enhancer.enhance(parameters):
                    monitor.update_progress(output.progress, progress_goal)
                    monitor.update_enhanced(self.item, output.dataset)
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
        except Exception:
            # Foreground state update runs even on failure so the UI reflects
            # FAILED before the exception propagates through TaskManager.
            self.task_monitor.transition_item_state(self.item, FluorescenceItemState.FAILED)
            raise
        else:
            # Natural completion or user stop both land here — user-stop is
            # considered a graceful end of the enhancement, not a failure.
            self.task_monitor.transition_item_state(self.item, FluorescenceItemState.READY)
