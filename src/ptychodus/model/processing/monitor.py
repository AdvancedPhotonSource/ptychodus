from __future__ import annotations
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import logging
import time

from pydantic import BaseModel

from ptychodus.api.io import save_product
from ptychodus.api.product import Product
from ptychodus.api.reconstruct import (
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
    TrainableReconstructor,
)
from ptychodus.api.spawn import iterate_in_subprocess

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
    """Run one reconstruction in a spawned child restricted to the selected devices.

    The child gives each run a clean GPU context, released when it exits, so
    backends built on different frameworks never share driver state with each
    other or with this process. ``device_environment`` is asked when the run
    starts, so the selection in force then is the one applied.
    """

    task_monitor: ProcessingTaskMonitor
    reconstructor: Reconstructor
    reconstruct_input: ReconstructInput
    product_item: ProductRepositoryItem
    output_product_file: Path | None
    options: BaseModel
    device_environment: Callable[[], Mapping[str, str]]

    def __call__(self) -> None:
        with self.task_monitor as monitor:
            # Zero until the run says otherwise, which a progress bar reads
            # as indeterminate: only the run knows how many steps it will take.
            monitor.update_progress(0, 0)
            tic = time.perf_counter()
            work = partial(
                self.reconstructor.reconstruct, self.reconstruct_input, options=self.options
            )
            results = iterate_in_subprocess(work, environment=self.device_environment())

            # Closed on the way out, so a stop tears the child down here rather
            # than whenever the abandoned iterator is collected.
            with closing(results):
                for result in results:
                    monitor.update_progress(result.progress, result.progress_goal)
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
    """Run one training session in a spawned child, then load the model it reports.

    Runs as :class:`ReconstructBackgroundTask` does. Training does not load
    what it wrote, and could not usefully do so in a child that is gone when it
    returns, so the model a finished run names in ``trained_model`` is loaded
    here. A stopped run loads nothing, so a half-written model is never picked up.
    """

    task_monitor: ProcessingTaskMonitor
    trainer: TrainableReconstructor
    reconstruct_input: ReconstructInput
    input_directory: Path
    output_directory: Path
    options: BaseModel
    device_environment: Callable[[], Mapping[str, str]]

    def __call__(self) -> None:
        with self.task_monitor as monitor:
            monitor.update_progress(0, 0)
            tic = time.perf_counter()
            trained_model: Path | None = None
            work = partial(
                self.trainer.train,
                self.input_directory,
                self.output_directory,
                options=self.options,
            )
            results = iterate_in_subprocess(work, environment=self.device_environment())

            with closing(results):
                for result in results:
                    monitor.update_progress(result.progress, result.progress_goal)

                    if result.trained_model is not None:
                        trained_model = result.trained_model

                    if monitor.is_stopping:
                        break

            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds.')

            if monitor.is_stopping:
                return

            if trained_model is None:
                logger.warning(f'{self.trainer.name} finished without reporting a trained model.')
            else:
                self.trainer.load_model(trained_model)
