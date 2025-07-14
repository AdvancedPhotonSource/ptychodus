from collections.abc import Sequence
from pathlib import Path
import logging
import time

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    ReconstructInput,
    Reconstructor,
    TrainOutput,
    TrainableReconstructor,
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

    def get_reconstruct_input(
        self,
        input_product_index: int,
        *,
        index_filter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> ReconstructInput:
        return self._data_matcher.match_diffraction_patterns_with_positions(
            input_product_index, index_filter=index_filter
        )

    def reconstruct(
        self,
        input_product_index: int,
        *,
        output_product_suffix: str = '',
        transform: int | None = None,
        index_filter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> int:
        reconstructor = self._reconstructor_chooser.get_current_plugin()
        input_product_item = self._product_api.get_item(input_product_index)
        output_product_index = self._product_api.insert_product(input_product_item.get_product())
        output_product_item = self._product_api.get_item(output_product_index)
        output_product_name = f'{input_product_item.get_name()}_{reconstructor.simple_name}'

        if output_product_suffix:
            output_product_name += f'_{output_product_suffix}'

        output_product_item.set_name(output_product_name)

        if transform is not None:
            scan_item_transform = output_product_item.get_scan_item().get_transform()
            scan_item_transform.apply_presets(transform)

            object_item = output_product_item.get_object_item()
            object_item.rebuild(recenter=True)

        self._reconstruction_queue.put(reconstructor.strategy, output_product_index, index_filter)
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

    def reconstruct_transformed(self, input_product_index: int) -> Sequence[int]:
        output_product_indexes: list[int] = []
        input_product = self._product_api.get_item(input_product_index)

        for preset_value, preset_label in enumerate(
            input_product.get_scan_item().get_transform().labels_for_presets()
        ):
            output_product_index = self.reconstruct(
                input_product_index,
                output_product_suffix=preset_label,
                transform=preset_value,
                index_filter=ScanIndexFilter.ALL,
            )
            output_product_indexes.append(output_product_index)

        return output_product_indexes

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
        result = TrainOutput([], [], -1)

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Training...')
            tic = time.perf_counter()
            result = reconstructor.train(data_path)
            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds. (code={result.result})')
        else:
            logger.warning('Reconstructor is not trainable!')

        return result

    def set_reconstructor(self, name: str) -> str:
        self._reconstructor_chooser.set_current_plugin(name)
        return self._reconstructor_chooser.get_current_plugin().simple_name
