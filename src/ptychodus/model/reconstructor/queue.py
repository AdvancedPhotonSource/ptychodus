from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import queue
import threading
import time

from ptychodus.api.reconstructor import Reconstructor, ReconstructOutput

from ..product import ProductRepositoryItem
from .matcher import DiffractionPatternPositionMatcher, ScanIndexFilter

logger = logging.getLogger(__name__)

__all__ = ['ReconstructionQueue']


class ReconstructionTask(ABC):
    @abstractmethod
    def execute(self) -> ReconstructionTask | None:
        pass


class UpdateProductTask(ReconstructionTask):
    def __init__(self, product: ProductRepositoryItem, result: ReconstructOutput) -> None:
        self._product = product
        self._result = result

    def execute(self) -> None:
        name = self._product.get_name()
        self._product.assign(self._result.product)
        self._product.set_name(name)


class ExecuteReconstructorTask(ReconstructionTask):
    def __init__(
        self,
        data_matcher: DiffractionPatternPositionMatcher,
        reconstructor: Reconstructor,
        product_index: int,
        index_filter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> None:
        self._data_matcher = data_matcher
        self._reconstructor = reconstructor
        self._product_index = product_index
        self._index_filter = index_filter

    def execute(self) -> UpdateProductTask:
        product_item = self._data_matcher.get_product_item(self._product_index)
        logger.info(f'Reconstructing {product_item.get_name()}...')

        logger.info('Preparing input data...')
        tic = time.perf_counter()
        parameters = self._data_matcher.match_diffraction_patterns_with_positions(
            self._product_index, self._index_filter
        )
        toc = time.perf_counter()
        logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

        logger.debug(parameters)
        tic = time.perf_counter()
        result = self._reconstructor.reconstruct(parameters)
        toc = time.perf_counter()
        logger.info(f'Reconstruction time {toc - tic:.4f} seconds. (code={result.result})')

        return UpdateProductTask(product_item, result)


class ReconstructionQueue:
    def __init__(self, data_matcher: DiffractionPatternPositionMatcher) -> None:
        self._data_matcher = data_matcher
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
                    logger.debug('Reconstructing...')
                    output_task = input_task.execute()
                    logger.debug('Reconstruction finished.')
                except Exception:
                    logger.exception('Reconstructor error!')
                else:
                    logger.debug('Adding reconstruction result to output queue...')
                    self._output_queue.put(output_task)
                    logger.debug('Reconstruction result added to output queue.')
                finally:
                    self._input_queue.task_done()
            except queue.Empty:
                pass

    def put(
        self,
        reconstructor: Reconstructor,
        product_index: int,
        index_filter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> None:
        task = ExecuteReconstructorTask(
            self._data_matcher, reconstructor, product_index, index_filter
        )
        logger.debug('Adding reconstruction task to queue...')
        self._input_queue.put(task)
        logger.debug('Reconstruction task added to queue.')

    def process_results(self, *, block: bool) -> None:
        while True:
            try:
                task = self._output_queue.get(block=block)

                try:
                    logger.debug('Processing reconstruction result...')
                    task.execute()
                    logger.debug('Reconstruction result processed.')
                finally:
                    self._output_queue.task_done()

                    if block and self._output_queue.empty():
                        logger.debug('No more results to process.')
                        break
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
