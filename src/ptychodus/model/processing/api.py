from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import logging
import time

from pydantic import BaseModel

from ptychodus.api.reconstruct import (
    NullReconstructor,
    PositionIndexFilter,
    ReconstructInput,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
    prepare_reconstruct_input,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import Parameter, StringParameter

from ..naming import split_name_counter
from ..product import ProductAPI
from ..task_manager import TaskManager
from .parameter_store import ProcessingParameterStore
from .monitor import (
    ProcessingTaskMonitor,
    ReconstructBackgroundTask,
    TrainBackgroundTask,
)

logger = logging.getLogger(__name__)


def _build_output_product_name(input_name: str, algorithm: str, suffix: str) -> str:
    """Name a reconstruction output after its input product.

    The collision counter is stripped first so that continuing a reconstruction from an
    earlier output recognizes the algorithm tag it already carries and lets the counter
    advance, instead of appending the same tag again. The odd/even suffix is peeled and
    re-applied for the same reason.
    """
    stem, _ = split_name_counter(input_name)

    if suffix and stem.endswith(f'_{suffix}'):
        stem = stem[: -len(suffix) - 1]

    if not stem.endswith(f'_{algorithm}'):
        stem += f'_{algorithm}'

    return f'{stem}_{suffix}' if suffix else stem


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
                # The persisted value is left alone rather than written back.
                # A settings file naming a backend this install lacks is an
                # ordinary thing to carry between machines, and rewriting the
                # key would lose the user's choice the first time it was
                # opened somewhere that could not resolve it.
                logger.warning(
                    f'Invalid plugin name "{value}". '
                    f'Registered plugins: {sorted(self._reconstructor_map)}.'
                )
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


def validate_options(reconstructor: Reconstructor, options: BaseModel) -> None:
    """Reject options that are not of the type *reconstructor* declared.

    ``options`` is typed as the base model, so nothing but this check ties a
    reconstructor to the options it asked for.
    """
    declared = type(reconstructor.get_default_options())

    if not isinstance(options, declared):
        raise TypeError(
            f'{reconstructor.name} declares {declared.__name__} '
            f'but received {type(options).__name__}.'
        )


class ProcessingAPI:
    def __init__(
        self,
        task_manager: TaskManager,
        product_api: ProductAPI,
        algorithm: ProcessingAlgorithmParameter,
        task_monitor: ProcessingTaskMonitor,
        parameter_store: ProcessingParameterStore,
    ) -> None:
        self._task_manager = task_manager
        self._product_api = product_api
        self._algorithm_parameter = algorithm
        self._task_monitor = task_monitor
        self._parameter_store = parameter_store

    def _options_for(self, reconstructor: Reconstructor) -> BaseModel:
        """Return the options to run *reconstructor* with, as currently edited.

        The store is keyed the same way the algorithm selection is, so the values
        served are the ones behind the panel the user was just looking at.
        """
        options = self._parameter_store.get_options(self._algorithm_parameter.get_value())
        validate_options(reconstructor, options)
        return options

    @property
    def task_monitor(self) -> ProcessingTaskMonitor:
        return self._task_monitor

    def get_reconstruct_input(
        self,
        *,
        product_index: int,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
    ) -> ReconstructInput:
        product_item = self._product_api.get_item(product_index)
        dataset = product_item.get_dataset()

        if dataset is None:
            raise RuntimeError(
                f'Product "{product_item.get_name()}" has no associated diffraction dataset.'
            )

        product = product_item.get_product()
        logger.info(f'Preparing input data for {product.metadata.name}...')
        tic = time.perf_counter()
        assembled_data = dataset.get_assembled_data()
        reconstruct_input = prepare_reconstruct_input(
            assembled_data, product, index_filter=index_filter
        )
        toc = time.perf_counter()
        logger.info(f'Data preparation time {toc - tic:.4f} seconds.')
        return reconstruct_input

    def reconstruct(
        self,
        *,
        product_index: int,
        algorithm: str | None = None,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
        output_product_suffix: str = '',
        output_product_file: Path | None = None,
        block: bool = False,
    ) -> int:
        self.set_reconstructor_if_provided(algorithm)
        input_product_item = self._product_api.get_item(product_index)
        output_product_name = _build_output_product_name(
            input_product_item.get_name(),
            self._algorithm_parameter.get_value(),
            output_product_suffix,
        )
        output_product_index = self._product_api.insert_product(
            input_product_item.get_product(),
            name=output_product_name,
            dataset=input_product_item.get_dataset(),
        )
        output_product_item = self._product_api.get_item(output_product_index)
        reconstruct_input = self.get_reconstruct_input(
            product_index=output_product_index,
            index_filter=index_filter,
        )
        reconstructor = self._algorithm_parameter.get_current_reconstructor()
        options = self._options_for(reconstructor)
        background_task = ReconstructBackgroundTask(
            self._task_monitor,
            reconstructor,
            reconstruct_input,
            output_product_item,
            output_product_file,
            options,
        )
        snapshot = self._task_monitor.get_completed_runs()
        self._task_manager.put_background_task(background_task)

        if block:
            while not self._task_manager.is_stopping:
                if self._task_monitor.wait_for_completion_after(
                    snapshot, timeout=TaskManager.WAIT_TIME_S
                ):
                    self._task_manager.run_foreground_tasks()
                    break
            self._task_monitor.raise_if_failed()

        return output_product_index

    def reconstruct_split(
        self, *, product_index: int, algorithm: str | None = None
    ) -> tuple[int, int]:
        output_product_index_odd = self.reconstruct(
            product_index=product_index,
            algorithm=algorithm,
            index_filter=PositionIndexFilter.ODD,
            output_product_suffix='odd',
        )
        output_product_index_even = self.reconstruct(
            product_index=product_index,
            algorithm=algorithm,
            index_filter=PositionIndexFilter.EVEN,
            output_product_suffix='even',
        )

        return output_product_index_odd, output_product_index_even

    def load_model(self, file_path: Path, algorithm: str | None = None) -> None:
        self.set_reconstructor_if_provided(algorithm)
        reconstructor = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening model...')
            tic = time.perf_counter()
            reconstructor.load_model(file_path)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Algorithm is not trainable!')

    def save_model(self, file_path: Path, algorithm: str | None = None) -> None:
        self.set_reconstructor_if_provided(algorithm)
        reconstructor = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Saving model...')
            tic = time.perf_counter()
            reconstructor.save_model(file_path)
            toc = time.perf_counter()
            logger.info(f'Save time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Algorithm is not trainable!')

    def export_training_data(
        self,
        file_path: Path,
        *,
        product_index: int,
        algorithm: str | None = None,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
    ) -> None:
        self.set_reconstructor_if_provided(algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            reconstruct_input = self.get_reconstruct_input(
                product_index=product_index,
                index_filter=index_filter,
            )

            logger.info('Exporting...')
            tic = time.perf_counter()
            options = self._options_for(trainer)
            trainer.export_training_data(file_path, reconstruct_input, options=options)
            toc = time.perf_counter()
            logger.info(f'Export time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Algorithm is not trainable!')

    def train(
        self,
        input_directory: Path,
        output_directory: Path,
        *,
        product_index: int,
        algorithm: str | None = None,
        block: bool = False,
    ) -> None:
        self.set_reconstructor_if_provided(algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            reconstruct_input = self.get_reconstruct_input(product_index=product_index)
            options = self._options_for(trainer)
            background_task = TrainBackgroundTask(
                self._task_monitor,
                trainer,
                reconstruct_input,
                input_directory,
                output_directory,
                options,
            )
            snapshot = self._task_monitor.get_completed_runs()
            self._task_manager.put_background_task(background_task)

            if block:
                while not self._task_manager.is_stopping:
                    if self._task_monitor.wait_for_completion_after(
                        snapshot, timeout=TaskManager.WAIT_TIME_S
                    ):
                        self._task_manager.run_foreground_tasks()
                        break
                self._task_monitor.raise_if_failed()
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

    def get_model_file_extension(self, *, algorithm: str | None = None) -> str:
        self.set_reconstructor_if_provided(algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            return trainer.get_model_file_extension()
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
