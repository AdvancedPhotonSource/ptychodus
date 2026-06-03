from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import time

from ptychodus.api.io import save_product
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
    TrainableReconstructor,
)

from ..product import ProductRepositoryItem
from ..task_monitor import TaskProgressMonitor

__all__ = [
    'ProcessingTaskMonitor',
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


class ProcessingTaskMonitor(TaskProgressMonitor):
    def update_product(
        self, product_item: ProductRepositoryItem, result: ReconstructOutput
    ) -> None:
        # TODO log handler messages to product comments
        task = UpdateProductTask(product_item, result.product)
        self._foreground_task_manager.put_foreground_task(task)


@dataclass(frozen=True)
class ReconstructBackgroundTask:
    task_monitor: ProcessingTaskMonitor
    reconstructor: Reconstructor
    reconstruct_input: ReconstructInput
    product_item: ProductRepositoryItem
    output_product_file: Path | None

    def __call__(self) -> None:
        with self.task_monitor as monitor:
            monitor.update_progress(0, self.reconstructor.get_progress_goal())
            tic = time.perf_counter()

            for result in self.reconstructor.reconstruct(self.reconstruct_input):
                monitor.update_progress(result.progress, self.reconstructor.get_progress_goal())
                monitor.update_product(self.product_item, result)
                of = self.output_product_file

                if of is not None:
                    of = of.parent / f'{of.stem}.{result.progress:06d}{of.suffix}'
                    save_product(of, result.product)

                if monitor.is_stopping:
                    break

            toc = time.perf_counter()
            logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')


@dataclass(frozen=True)
class TrainBackgroundTask:
    task_monitor: ProcessingTaskMonitor
    trainer: TrainableReconstructor
    reconstruct_input: ReconstructInput
    input_path: Path
    output_path: Path

    def __call__(self) -> None:
        with self.task_monitor as monitor:
            monitor.update_progress(0, self.trainer.get_progress_goal())
            tic = time.perf_counter()

            for result in self.trainer.train(self.input_path, self.output_path):
                monitor.update_progress(result.progress, self.trainer.get_progress_goal())

                if monitor.is_stopping:
                    break

            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds.')
