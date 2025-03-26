from pathlib import Path
import logging
import time

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    Reconstructor,
    TrainableReconstructor,
    TrainOutput,
)

from ..product import ProductAPI
from .matcher import DiffractionPatternPositionMatcher, ScanIndexFilter
from .queue import ReconstructionQueue

logger = logging.getLogger(__name__)


class ReconstructorAPI:
    def __init__(
        self,
        reconstruction_queue: ReconstructionQueue,
        data_matcher: DiffractionPatternPositionMatcher,
        product_api: ProductAPI,
        reconstructor_chooser: PluginChooser[Reconstructor],
    ) -> None:
        self._reconstruction_queue = reconstruction_queue
        self._data_matcher = data_matcher
        self._product_api = product_api
        self._reconstructor_chooser = reconstructor_chooser

    @property
    def is_reconstructing(self) -> bool:
        return self._reconstruction_queue.is_reconstructing

    def process_results(self, *, block: bool) -> None:
        self._reconstruction_queue.process_results(block=block)

    def reconstruct(
        self,
        input_product_index: int,
        *,
        output_product_suffix: str = '',
        index_filter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> int:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy
        parameters = self._data_matcher.match_diffraction_patterns_with_positions(
            input_product_index, index_filter
        )
        output_product_index = self._product_api.insert_new_product(
            like_index=input_product_index, mutable=False
        )
        output_product = self._product_api.get_item(output_product_index)

        output_product_name = (
            self._data_matcher.get_product_name(input_product_index)
            + f'_{self._reconstructor_chooser.get_current_plugin().simple_name}'
        )

        if output_product_suffix:
            output_product_name += f'_{output_product_suffix}'

        output_product.set_name(output_product_name)
        self._reconstruction_queue.put(reconstructor, parameters, output_product)
        return output_product_index

    def reconstruct_split(self, input_product_index: int) -> tuple[int, int]:
        output_product_index_odd = self.reconstruct(
            input_product_index,
            output_product_suffix='odd',
            index_filter=ScanIndexFilter.ODD,
        )
        output_product_index_even = self.reconstruct(
            input_product_index,
            output_product_suffix='even',
            index_filter=ScanIndexFilter.EVEN,
        )

        return output_product_index_odd, output_product_index_even

    def open_model(self, file_path: Path) -> None:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening model...')
            tic = time.perf_counter()
            reconstructor.open_model(file_path)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def save_model(self, file_path: Path) -> None:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Saving model...')
            tic = time.perf_counter()
            reconstructor.save_model(file_path)
            toc = time.perf_counter()
            logger.info(f'Save time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def export_training_data(self, file_path: Path, input_product_index: int) -> None:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Preparing input data...')
            tic = time.perf_counter()
            parameters = self._data_matcher.match_diffraction_patterns_with_positions(
                input_product_index, ScanIndexFilter.ALL
            )
            toc = time.perf_counter()
            logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

            logger.info('Exporting...')
            tic = time.perf_counter()
            reconstructor.export_training_data(file_path, parameters)
            toc = time.perf_counter()
            logger.info(f'Export time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def train(self, data_path: Path) -> TrainOutput:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy
        result = TrainOutput([], -1)

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Training...')
            tic = time.perf_counter()
            result = reconstructor.train(data_path)
            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds. (code={result.result})')
        else:
            logger.warning('Reconstructor is not trainable!')

        return result
