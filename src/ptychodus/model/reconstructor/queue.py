from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import queue
import threading
import time

from ptychodus.api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput

from ..product import ProductRepositoryItem

logger = logging.getLogger(__name__)

__all__ = ['ReconstructionQueue']


class ReconstructionTask(ABC):
    @abstractmethod
    def execute(self) -> ReconstructionTask | None:
        pass


class UpdateProductTask(ReconstructionTask):
    def __init__(self, result: ReconstructOutput, product: ProductRepositoryItem) -> None:
        self._result = result
        self._product = product

    def execute(self) -> None:
        name = self._product.get_name()
        self._product.assign(self._result.product)
        self._product.set_name(name)


class ExecuteReconstructorTask(ReconstructionTask):
    def __init__(
        self,
        reconstructor: Reconstructor,
        parameters: ReconstructInput,
        product: ProductRepositoryItem,
    ) -> None:
        self._reconstructor = reconstructor
        self._parameters = parameters
        self._product = product

    def execute(self) -> UpdateProductTask:
        tic = time.perf_counter()
        result = self._reconstructor.reconstruct(self._parameters)
        toc = time.perf_counter()
        logger.info(f'Reconstruction time {toc - tic:.4f} seconds. (code={result.result})')
        return UpdateProductTask(result, self._product)


class ReconstructionQueue:
    def __init__(self) -> None:
        self._input_queue: queue.Queue[ExecuteReconstructorTask] = queue.Queue()
        self._output_queue: queue.Queue[UpdateProductTask] = queue.Queue()
        self._stop_work_event = threading.Event()
        self._worker = threading.Thread(target=self._reconstruct)

    @property
    def is_reconstructing(self) -> bool:
        return self._input_queue.unfinished_tasks > 0

    def _reconstruct(self) -> None:
        while not self._stop_work_event.is_set():
            try:
                input_task = self._input_queue.get(block=True, timeout=1)

                try:
                    output_task = input_task.execute()
                except Exception:
                    logger.exception('Reconstructor error!')
                else:
                    self._output_queue.put(output_task)
                finally:
                    self._input_queue.task_done()
            except queue.Empty:
                pass

    def put(
        self,
        reconstructor: Reconstructor,
        parameters: ReconstructInput,
        product: ProductRepositoryItem,
    ) -> None:
        task = ExecuteReconstructorTask(reconstructor, parameters, product)
        self._input_queue.put(task)

    def process_results(self, *, block: bool) -> None:
        while True:
            try:
                task = self._output_queue.get(block=block)

                try:
                    task.execute()
                finally:
                    self._output_queue.task_done()
            except queue.Empty:
                break

    def start(self) -> None:
        logger.info('Starting reconstructor...')
        self._worker.start()
        logger.info('Reconstructor started.')

    def stop(self) -> None:
        logger.info('Finishing reconstructions...')
        self._input_queue.join()

        logger.info('Stopping reconstructor...')
        self._stop_work_event.set()
        self._worker.join()
        self.process_results(block=False)
        logger.info('Reconstructor stopped.')

    def __len__(self) -> int:
        return self._input_queue.qsize()
