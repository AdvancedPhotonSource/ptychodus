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
    TrainableReconstructor,
)

from ..diffraction import DiffractionAPI
from ..product import ProductAPI
from ..task_manager import TaskManager
from .context import (
    ReconstructBackgroundTask,
    ProcessingContext,
    ProcessingProgressMonitor,
    TrainBackgroundTask,
)
from .settings import ProcessingSettings

logger = logging.getLogger(__name__)


class ProcessingAPI:
    def __init__(
        self,
        task_manager: TaskManager,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        settings: ProcessingSettings,
        context: ProcessingContext,
        algorithm_chooser: PluginChooser[Reconstructor],
    ) -> None:
        self._task_manager = task_manager
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self._context = context
        self._algorithm_chooser = algorithm_chooser

        algorithm_chooser.synchronize_with_parameter(settings.algorithm)

    def get_progress_monitor(self) -> ProcessingProgressMonitor:
        return self._context.get_progress_monitor()

    def get_reconstruct_input(
        self,
        product_index: int,
        *,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
    ) -> ReconstructInput:
        product = self._product_api.get_item(product_index).get_product()
        logger.info(f'Preparing input data for {product.metadata.name}...')
        tic = time.perf_counter()
        assembled_data = self._diffraction_api.get_assembled_data()
        reconstruct_input = assembled_data.prepare_reconstruct_input(
            product, index_filter=index_filter
        )
        toc = time.perf_counter()
        logger.info(f'Data preparation time {toc - tic:.4f} seconds.')
        return reconstruct_input

    def reconstruct(
        self,
        input_product_index: int,
        *,
        algorithm: str | None = None,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
        output_product_suffix: str = '',
        output_product_file: Path | None = None,
        block: bool = False,
    ) -> int:
        self.set_algorithm_if_provided(algorithm)
        algorithm_plugin = self._algorithm_chooser.get_current_plugin()
        input_product_item = self._product_api.get_item(input_product_index)
        output_product_index = self._product_api.insert_product(input_product_item.get_product())
        output_product_item = self._product_api.get_item(output_product_index)
        output_product_name = f'{input_product_item.get_name()}_{algorithm_plugin.simple_name}'

        if output_product_suffix:
            output_product_name += f'_{output_product_suffix}'

        output_product_item.set_name(output_product_name)
        reconstruct_input = self.get_reconstruct_input(
            output_product_index, index_filter=index_filter
        )
        finished_event = threading.Event()
        background_task = ReconstructBackgroundTask(
            self._context,
            algorithm_plugin.strategy,
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

    def load_model_from_file(self, file_path: Path, algorithm: str | None = None) -> None:
        self.set_algorithm_if_provided(algorithm)
        reconstructor = self._algorithm_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening model...')
            tic = time.perf_counter()
            reconstructor.load_model_from_file(file_path)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Algorithm is not trainable!')

    def export_training_data(
        self,
        file_path: Path,
        product_index: int,
        algorithm: str | None = None,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
    ) -> None:
        self.set_algorithm_if_provided(algorithm)
        trainer = self._algorithm_chooser.get_current_plugin().strategy

        if isinstance(trainer, TrainableReconstructor):
            reconstruct_input = self.get_reconstruct_input(product_index, index_filter=index_filter)

            logger.info('Exporting...')
            tic = time.perf_counter()
            trainer.export_training_data(file_path, reconstruct_input)
            toc = time.perf_counter()
            logger.info(f'Export time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Algorithm is not trainable!')

    def train(
        self,
        product_index: int,
        input_path: Path,
        output_path: Path,
        *,
        algorithm: str | None = None,
        block: bool = False,
    ) -> None:
        self.set_algorithm_if_provided(algorithm)
        algorithm_plugin = self._algorithm_chooser.get_current_plugin()
        trainer = algorithm_plugin.strategy

        if isinstance(trainer, TrainableReconstructor):
            reconstruct_input = self.get_reconstruct_input(product_index)
            finished_event = threading.Event()
            background_task = TrainBackgroundTask(
                self._context,
                trainer,
                reconstruct_input,
                finished_event,
                input_path,
                output_path,
            )
            self._task_manager.put_background_task(background_task)

            if block:
                while not self._task_manager.is_stopping:
                    if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                        self._task_manager.run_foreground_tasks()
                        break
        else:
            logger.warning('Algorithm is not trainable!')

    def available_reconstructors(self, *, trainable: bool) -> Iterator[str]:
        if trainable:
            for plugin in self._algorithm_chooser:
                if isinstance(plugin.strategy, TrainableReconstructor):
                    yield plugin.display_name
        else:
            for plugin in self._algorithm_chooser:
                if not isinstance(plugin.strategy, TrainableReconstructor):
                    yield plugin.display_name

    def set_algorithm_if_provided(self, algorithm: str | None) -> None:
        if algorithm is not None:
            self._algorithm_chooser.set_current_plugin(algorithm)
