from collections.abc import Iterator
from pathlib import Path
import logging
import threading
import time

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    PositionIndexFilter,
    ReconstructInput,
    Reconstructor,
    TrainOutput,
    TrainableReconstructor,
)

from ..diffraction import AssembledDiffractionDataset
from ..product import ProductAPI
from ..task_manager import TaskManager
from .context import ReconstructBackgroundTask, ProcessingContext, ProcessingProgressMonitor
from .settings import ProcessingSettings

logger = logging.getLogger(__name__)


class ProcessingAPI:
    def __init__(
        self,
        task_manager: TaskManager,
        diffraction_dataset: AssembledDiffractionDataset,
        product_api: ProductAPI,
        settings: ProcessingSettings,
        context: ProcessingContext,
        algorithm_chooser: PluginChooser[Reconstructor],
    ) -> None:
        self._task_manager = task_manager
        self._diffraction_dataset = diffraction_dataset
        self._product_api = product_api
        self._context = context
        self._algorithm_chooser = algorithm_chooser

        algorithm_chooser.synchronize_with_parameter(settings.algorithm)

    def get_progress_monitor(self) -> ProcessingProgressMonitor:
        return self._context.get_progress_monitor()

    def get_reconstruct_input(
        self,
        input_product_index: int,
        *,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
    ) -> ReconstructInput:
        input_product = self._product_api.get_item(input_product_index).get_product()
        assembled_data = self._diffraction_dataset.get_assembled_data()
        return assembled_data.prepare_reconstruct_input(input_product, index_filter=index_filter)

    def reconstruct(
        self,
        input_product_index: int,
        *,
        algorithm: str | None = None,
        output_product_suffix: str = '',
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
        output_product_file: Path | None = None,
        block: bool = False,
    ) -> int:
        reconstructor = self._algorithm_chooser.get_current_plugin()  # FIXME algorithm
        input_product_item = self._product_api.get_item(input_product_index)
        output_product_index = self._product_api.insert_product(input_product_item.get_product())
        output_product_item = self._product_api.get_item(output_product_index)
        output_product_name = f'{input_product_item.get_name()}_{reconstructor.simple_name}'

        if output_product_suffix:
            output_product_name += f'_{output_product_suffix}'

        output_product_item.set_name(output_product_name)

        logger.info(f'Preparing input data for {output_product_item.get_name()}...')
        tic = time.perf_counter()
        assembled_data = self._diffraction_dataset.get_assembled_data()
        reconstruct_input = assembled_data.prepare_reconstruct_input(
            output_product_item.get_product(), index_filter
        )
        toc = time.perf_counter()
        logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

        logger.debug(reconstruct_input)

        finished_event = threading.Event()

        background_task = ReconstructBackgroundTask(
            self._context,
            reconstructor.strategy,
            reconstruct_input,
            output_product_item,
            finished_event,
            output_product_file,
        )
        self._task_manager.put_background_task(background_task)

        if block:
            while not self._task_manager.is_stopping:
                if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                    self._task_manager.run_foreground_tasks()
                    break

        return output_product_index

    def reconstruct_split(self, input_product_index: int) -> tuple[int, int]:
        output_product_index_odd = self.reconstruct(
            input_product_index,
            output_product_suffix='odd',
            index_filter=PositionIndexFilter.ODD,
        )
        output_product_index_even = self.reconstruct(
            input_product_index,
            output_product_suffix='even',
            index_filter=PositionIndexFilter.EVEN,
        )

        return output_product_index_odd, output_product_index_even

    def open_model(self, file_path: Path) -> None:
        reconstructor = self._algorithm_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening model...')
            tic = time.perf_counter()
            reconstructor.open_model(file_path)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def save_model(self, file_path: Path) -> None:
        reconstructor = self._algorithm_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Saving model...')
            tic = time.perf_counter()
            reconstructor.save_model(file_path)
            toc = time.perf_counter()
            logger.info(f'Save time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def export_training_data(
        self, file_path: Path, input_product_index: int, algorithm: str | None = None
    ) -> None:
        # FIXME algorithm
        reconstructor = self._algorithm_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Preparing input data...')
            tic = time.perf_counter()
            reconstruct_input = self.get_reconstruct_input(input_product_index)
            toc = time.perf_counter()
            logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

            logger.info('Exporting...')
            tic = time.perf_counter()
            reconstructor.export_training_data(file_path, reconstruct_input)
            toc = time.perf_counter()
            logger.info(f'Export time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def train(
        self,
        input_product_index: int,
        data_path: Path,
        *,
        algorithm: str | None = None,
        block: bool = False,
    ) -> TrainOutput:
        # FIXME input_product_index
        # FIXME algorithm
        # FIXME block
        reconstructor = self._algorithm_chooser.get_current_plugin().strategy
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

    def available_reconstructors(self, *, trainable: bool) -> Iterator[str]:
        # FIXME trainable
        for plugin in self._algorithm_chooser:
            yield plugin.display_name

    def set_reconstructor(self, name: str) -> str:  # FIXME remove
        self._algorithm_chooser.set_current_plugin(name)
        return self._algorithm_chooser.get_current_plugin().simple_name
