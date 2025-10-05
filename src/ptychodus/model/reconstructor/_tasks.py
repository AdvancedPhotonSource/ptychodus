from __future__ import annotations
import logging
import threading
import time

from ptychodus.api.product import Product
from ptychodus.api.reconstructor import Reconstructor, ReconstructOutput

from ..product import ProductRepositoryItem
from .matcher import DiffractionPatternPositionMatcher, ScanIndexFilter
from ..task_manager import ForegroundTask, ForegroundTaskManager

logger = logging.getLogger(__name__)


class UpdateProductTask:
    def __init__(self, product_item: ProductRepositoryItem, product: Product) -> None:
        self._product_item = product_item
        self._product = product

    def __call__(self) -> None:
        name = self._product_item.get_name()
        self._product_item.assign(self._product)
        self._product_item.set_name(name)


class ReconstructTask:
    def __init__(
        self,
        is_reconstructing: threading.Event,
        foreground_task_manager: ForegroundTaskManager,
        data_matcher: DiffractionPatternPositionMatcher,
        reconstructor: Reconstructor,
        product_index: int,
        index_filter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> None:
        self._is_reconstructing = is_reconstructing
        self._foreground_task_manager = foreground_task_manager
        self._data_matcher = data_matcher
        self._reconstructor = reconstructor
        self._product_index = product_index
        self._index_filter = index_filter

    def _update_object(
        self, product_item: ProductRepositoryItem, result: ReconstructOutput
    ) -> None:
        task = UpdateProductTask(product_item, result.product)
        self._foreground_task_manager.put_foreground_task(task)

    def __call__(self) -> ForegroundTask | None:
        self._is_reconstructing.set()

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
        # FIXME Use spawn or forkserver, which ensures a clean initialization of each process.
        # FIXME Explicitly set the start method using torch.multiprocessing.set_start_method('spawn', force=True) or 'forkserver'.
        #       This should be done at the beginning of your program, typically within an if __name__ == "__main__": block.
        result = self._reconstructor.reconstruct(parameters)  # FIXME _update_object
        toc = time.perf_counter()
        logger.info(f'Reconstruction time {toc - tic:.4f} seconds. (code={result.result})')

        self._is_reconstructing.clear()

        return UpdateProductTask(product_item, result.product)


class TrainTask:  # TODO
    def __call__(self) -> ForegroundTask | None:
        pass
