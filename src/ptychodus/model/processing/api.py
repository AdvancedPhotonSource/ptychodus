from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import logging
import threading
import time

from ptychodus.api.reconstructor import (
    NullReconstructor,
    PositionIndexFilter,
    ReconstructInput,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter, StringParameter

from ..diffraction import DiffractionAPI
from ..product import ProductAPI
from ..task_manager import TaskManager
from .context import (
    ProcessingContext,
    ProcessingProgressMonitor,
    ReconstructBackgroundTask,
    TrainBackgroundTask,
)

logger = logging.getLogger(__name__)


class ProcessingAlgorithmParameter(Parameter[str], Observer):
    @staticmethod
    def generate_key(library: str, reconstructor: str) -> str:
        return f'{library.casefold()}_{reconstructor.casefold()}'

    @staticmethod
    def split_key(key: str) -> tuple[str, str]:
        library, reconstructor = key.split('_', maxsplit=1)
        return library, reconstructor

    def __init__(
        self, parameter: StringParameter, libraries: Sequence[ReconstructorLibrary]
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._libraries = libraries
        self._reconstructor_map: Mapping[str, Reconstructor] = {
            self.generate_key(library.name, reconstructor.name): reconstructor
            for library in libraries
            for reconstructor in library
        }
        self._current_value = 'none_none'
        self._current_reconstructor: Reconstructor = NullReconstructor('None')

        parameter.add_observer(self)
        self.set_value(parameter.get_value())

    def get_value(self) -> str:
        return self._current_value

    def set_value(self, value: str, *, notify: bool = True) -> None:
        sanitized_value = value.strip().casefold()

        if sanitized_value != self._current_value:
            try:
                reconstructor = self._reconstructor_map[sanitized_value]
            except KeyError:
                logger.debug(
                    f'Invalid plugin name "{value}". '
                    f'Registered plugins: {self._reconstructor_map.keys()}.'
                )
                self._parameter.set_value(self._current_value)
            else:
                logger.debug(f'{self._current_value} -> {sanitized_value}')
                self._current_value = sanitized_value
                self._current_reconstructor = reconstructor
                self._parameter.set_value(self._current_value)

                if notify:
                    self.notify_observers()

    def get_value_as_string(self) -> str:
        return self.get_value()

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value)

    def available_reconstructors(self) -> Iterator[str]:
        for key in self._reconstructor_map.keys():
            yield key

    def get_current_reconstructor(self) -> Reconstructor:
        return self._current_reconstructor

    def copy(self) -> Parameter[str]:
        return ProcessingAlgorithmParameter(self._parameter, self._libraries)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.set_value(self._parameter.get_value())


class ProcessingAPI:
    def __init__(
        self,
        task_manager: TaskManager,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        algorithm: ProcessingAlgorithmParameter,
        context: ProcessingContext,
    ) -> None:
        self._task_manager = task_manager
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self._algorithm_parameter = algorithm
        self._context = context

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
        self.set_reconstructor_if_provided(algorithm)
        input_product_item = self._product_api.get_item(input_product_index)
        output_product_index = self._product_api.insert_product(input_product_item.get_product())
        output_product_item = self._product_api.get_item(output_product_index)
        output_product_name = (
            f'{input_product_item.get_name()}_{self._algorithm_parameter.get_value()}'
        )

        if output_product_suffix:
            output_product_name += f'_{output_product_suffix}'

        output_product_item.set_name(output_product_name)
        reconstruct_input = self.get_reconstruct_input(
            output_product_index, index_filter=index_filter
        )
        finished_event = threading.Event()
        background_task = ReconstructBackgroundTask(
            self._context,
            self._algorithm_parameter.get_current_reconstructor(),
            reconstruct_input,
            output_product_item,
            output_product_file,
            finished_event,
        )
        self._task_manager.put_background_task(background_task)

        if block:
            while not self._task_manager.is_stopping:
                if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                    self._task_manager.run_foreground_tasks()
                    break

        return output_product_index

    def reconstruct_split(
        self, input_product_index: int, *, algorithm: str | None = None
    ) -> tuple[int, int]:
        output_product_index_odd = self.reconstruct(
            input_product_index,
            algorithm=algorithm,
            index_filter=PositionIndexFilter.ODD,
            output_product_suffix='odd',
        )
        output_product_index_even = self.reconstruct(
            input_product_index,
            algorithm=algorithm,
            index_filter=PositionIndexFilter.EVEN,
            output_product_suffix='even',
        )

        return output_product_index_odd, output_product_index_even

    def load_model_from_file(self, file_path: Path, algorithm: str | None = None) -> None:
        self.set_reconstructor_if_provided(algorithm)
        reconstructor = self._algorithm_parameter.get_current_reconstructor()

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
        self.set_reconstructor_if_provided(algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

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
        self.set_reconstructor_if_provided(algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            reconstruct_input = self.get_reconstruct_input(product_index)
            finished_event = threading.Event()
            background_task = TrainBackgroundTask(
                self._context,
                trainer,
                reconstruct_input,
                input_path,
                output_path,
                finished_event,
            )
            self._task_manager.put_background_task(background_task)

            if block:
                while not self._task_manager.is_stopping:
                    if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                        self._task_manager.run_foreground_tasks()
                        break
        else:
            logger.warning('Algorithm is not trainable!')

    def get_training_data_file_filter(self, *, algorithm: str | None = None) -> str:
        self.set_reconstructor_if_provided(algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            return trainer.get_training_data_file_filter()
        else:
            logger.warning('Algorithm is not trainable!')

        return ''

    def get_model_file_filter(self, *, algorithm: str | None = None) -> str:
        self.set_reconstructor_if_provided(algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            return trainer.get_model_file_filter()
        else:
            logger.warning('Algorithm is not trainable!')

        return ''

    def available_reconstructors(self) -> Iterator[str]:
        return self._algorithm_parameter.available_reconstructors()

    def available_reconstructors_parts(self) -> Iterator[tuple[str, str]]:
        for key in self._algorithm_parameter.available_reconstructors():
            yield self._algorithm_parameter.split_key(key)

    def set_reconstructor_if_provided(self, algorithm: str | None) -> None:
        if algorithm is not None:
            self._algorithm_parameter.set_value(algorithm)

    def is_reconstructor_trainable(self) -> bool:
        reconstructor = self._algorithm_parameter.get_current_reconstructor()
        return isinstance(reconstructor, TrainableReconstructor)
