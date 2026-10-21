from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
import logging
import time

from pydantic import BaseModel

from ptychodus.api.io import save_cxi_reconstruct_input
from ptychodus.api.reconstruct import (
    NullReconstructor,
    PositionIndexFilter,
    ReconstructInput,
    Reconstructor,
    ReconstructorLibrary,
    SubprocessReconstructor,
    TrainableReconstructor,
    prepare_reconstruct_input,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import Parameter, StringParameter
from ptychodus.api.plugins import PluginChooserParameter

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

_CXI_FILE_TYPE = 'CXI'


def _is_trainable(reconstructor: Reconstructor) -> bool:
    """Whether a reconstructor can train; a subprocess reconstructor says so itself."""
    if isinstance(reconstructor, SubprocessReconstructor):
        return reconstructor.is_trainable

    return isinstance(reconstructor, TrainableReconstructor)


def _algorithm_tag(library: str, algorithm: str) -> str:
    """Label a run by what produced it.

    One token, because the de-duplication below recognizes the tag a product
    already carries by matching the end of its name. Both halves, because two
    backends may offer an algorithm of the same name. An empty half is
    dropped, so an unresolved selection cannot leave a trailing separator.
    """
    return '_'.join(part.casefold() for part in (library, algorithm) if part)


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
    """Which reconstructor of the selected library a run uses.

    Scoped to the library beside it: the values this accepts are whatever
    that library offers, so changing the library changes what this means.
    Spelled as the backend spells it, resolved case-insensitively, and
    written back canonicalized.
    """

    def __init__(
        self,
        parameter: StringParameter,
        library: PluginChooserParameter[ReconstructorLibrary],
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._library = library
        self._current = ''
        self._current_reconstructor: Reconstructor = NullReconstructor('None')
        # The last algorithm chosen for each library this session, so that
        # looking at another backend and coming back does not quietly land
        # somewhere else. Not persisted: a key per library is more than a
        # settings file should carry for a convenience.
        self._remembered: dict[str, str] = {}

        parameter.add_observer(self)
        library.add_observer(self)
        self._settle(parameter.get_value(), persist=False)

    def _library_name(self) -> str:
        try:
            return self._library.get_value()
        except LookupError:
            # No backend registered at all, which only a test arranges.
            return ''

    def _available(self) -> Sequence[Reconstructor]:
        """What the selected library offers, or nothing when it offers nothing."""
        try:
            return list(self._library.get_strategy())
        except LookupError:
            return []

    def choices(self) -> Iterator[str]:
        """Yield every value this parameter accepts, as the backend spells it."""
        for reconstructor in self._available():
            yield reconstructor.name

    def _settle(self, wanted: str, *, persist: bool, warn: bool = True) -> None:
        """Resolve *wanted* against the selected library and select the result.

        An unresolvable name is stood in for by the first algorithm the
        library offers, so the application is usable -- but it is not written
        back unless *persist*. A settings file naming a backend or an
        algorithm this install lacks is an ordinary thing to carry between
        machines, and rewriting the key would lose the choice the first time
        it was opened somewhere that could not resolve it.

        *warn* distinguishes a name that was asked for from one merely
        carried over: changing library to one that does not offer the
        previous algorithm is an ordinary thing to do, not a complaint.
        """
        available = self._available()
        sanitized = wanted.strip().casefold()
        resolved = next((r for r in available if r.name.casefold() == sanitized), None)

        if resolved is None:
            if warn and sanitized and available:
                logger.warning(
                    f'Invalid algorithm "{wanted}" for {self._library_name()}. '
                    f'Available: {[r.name for r in available]}.'
                )

            selected = available[0] if available else None
        else:
            selected = resolved
            persist = True

        if selected is None:
            self._current = ''
            self._current_reconstructor = NullReconstructor('None')
        else:
            self._current = selected.name
            self._current_reconstructor = selected
            self._remembered[self._library_name()] = selected.name

            if persist:
                self._parameter.set_value(selected.name)

    def get_value(self) -> str:
        return self._current

    def set_value(self, value: str, *, notify: bool = True) -> None:
        if value.strip().casefold() == self._current.casefold():
            return

        self._settle(value, persist=True)

        if notify:
            self.notify_observers()

    def get_value_as_string(self) -> str:
        return self.get_value()

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value)

    def get_current_reconstructor(self) -> Reconstructor:
        return self._current_reconstructor

    def get_current_parts(self) -> tuple[str, str]:
        """The selected library and algorithm, as each is spelled."""
        return self._library_name(), self._current

    def copy(self) -> Parameter[str]:
        return ProcessingAlgorithmParameter(self._parameter, self._library)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._settle(self._parameter.get_value(), persist=False)
            self.notify_observers()
        elif observable is self._library:
            # Re-resolved from what is persisted rather than from what is
            # selected, because the two keys are applied in an order this
            # cannot control: a settings file is read after everything is
            # built, and the algorithm may land while the previous library is
            # still selected. Whatever settles is recorded, since changing
            # library is a choice rather than a load.
            wanted = self._remembered.get(self._library_name(), self._parameter.get_value())
            self._settle(wanted, persist=True, warn=False)
            self.notify_observers()


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
        library: PluginChooserParameter[ReconstructorLibrary],
        algorithm: ProcessingAlgorithmParameter,
        algorithms_by_library: Mapping[str, Sequence[str]],
        task_monitor: ProcessingTaskMonitor,
        parameter_store: ProcessingParameterStore,
    ) -> None:
        self._task_manager = task_manager
        self._product_api = product_api
        self._library_parameter = library
        self._algorithm_parameter = algorithm
        self._algorithms_by_library = {
            name.casefold(): tuple(algorithms) for name, algorithms in algorithms_by_library.items()
        }
        self._task_monitor = task_monitor
        self._parameter_store = parameter_store

    def _options_for(self, reconstructor: Reconstructor) -> BaseModel:
        """Return the options to run *reconstructor* with, as currently edited.

        The store is keyed the same way the algorithm selection is, so the values
        served are the ones behind the panel the user was just looking at.
        """
        library, algorithm = self._algorithm_parameter.get_current_parts()
        options = self._parameter_store.get_options(library, algorithm)
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

        carried: list[str] = []

        if reconstruct_input.background is not None:
            carried.append('background')

        if reconstruct_input.flat_field is not None:
            carried.append('flat field')

        if carried:
            logger.info(
                f'The patterns carry an unapplied {" and ".join(carried)}; a reconstructor'
                ' that does not model it ignores it. Enable the detector correction to'
                ' apply it to the patterns instead.'
            )

        return reconstruct_input

    def reconstruct(
        self,
        *,
        product_index: int,
        library: str | None = None,
        algorithm: str | None = None,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
        output_product_suffix: str = '',
        output_product_file: Path | None = None,
        block: bool = False,
    ) -> int:
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        input_product_item = self._product_api.get_item(product_index)
        output_product_name = _build_output_product_name(
            input_product_item.get_name(),
            _algorithm_tag(*self.current_selection()),
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
        self,
        *,
        product_index: int,
        library: str | None = None,
        algorithm: str | None = None,
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

    def load_model(
        self, file_path: Path, *, library: str | None = None, algorithm: str | None = None
    ) -> None:
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        reconstructor = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening model...')
            tic = time.perf_counter()
            reconstructor.load_model(file_path)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Algorithm is not trainable!')

    def save_model(
        self, file_path: Path, *, library: str | None = None, algorithm: str | None = None
    ) -> None:
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
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
        library: str | None = None,
        algorithm: str | None = None,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
        file_type: str | None = None,
    ) -> None:
        """Export what the current trainable reconstructor would train on.

        `file_type` of None writes the reconstructor's own format; ``'CXI'`` writes a
        standard CXI file holding the patterns, mask, detector maps and initial
        product, with a ``.cxi`` suffix added when the path has none.
        """
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if not _is_trainable(trainer):
            logger.warning('Algorithm is not trainable!')
            return

        if file_type is not None and file_type.casefold() != _CXI_FILE_TYPE.casefold():
            raise ValueError(f'Unknown training data file type "{file_type}".')

        reconstruct_input = self.get_reconstruct_input(
            product_index=product_index,
            index_filter=index_filter,
        )

        logger.info('Exporting...')
        tic = time.perf_counter()

        if file_type is None:
            assert isinstance(trainer, TrainableReconstructor)
            options = self._options_for(trainer)
            trainer.export_training_data(file_path, reconstruct_input, options=options)
        else:
            cxi_path = file_path if file_path.suffix else file_path.with_suffix('.cxi')
            save_cxi_reconstruct_input(cxi_path, reconstruct_input)

        toc = time.perf_counter()
        logger.info(f'Export time {toc - tic:.4f} seconds.')

    def train(
        self,
        input_directory: Path,
        output_directory: Path,
        *,
        product_index: int,
        library: str | None = None,
        algorithm: str | None = None,
        block: bool = False,
    ) -> None:
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
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

    def get_training_data_file_filters(
        self, *, library: str | None = None, algorithm: str | None = None
    ) -> dict[str, str | None]:
        """Map each file-dialog name filter to the `file_type` that exports it.

        The reconstructor's own format comes first, keyed to None, then CXI. A
        reconstructor that is not trainable offers nothing.
        """
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if not _is_trainable(trainer):
            logger.warning('Algorithm is not trainable!')
            return {}

        assert isinstance(trainer, TrainableReconstructor)
        filters: dict[str, str | None] = {}
        native_filter = trainer.get_training_data_file_filter()

        if native_filter:
            filters[native_filter] = None

        filters['Coherent X-ray Imaging Files (*.cxi)'] = _CXI_FILE_TYPE
        return filters

    def get_model_file_filter(
        self, *, library: str | None = None, algorithm: str | None = None
    ) -> str:
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            return trainer.get_model_file_filter()
        else:
            logger.warning('Algorithm is not trainable!')

        return ''

    def get_model_file_extension(
        self, *, library: str | None = None, algorithm: str | None = None
    ) -> str:
        self.set_reconstructor_if_provided(library=library, algorithm=algorithm)
        trainer = self._algorithm_parameter.get_current_reconstructor()

        if isinstance(trainer, TrainableReconstructor):
            return trainer.get_model_file_extension()
        else:
            logger.warning('Algorithm is not trainable!')

        return ''

    def available_libraries(self) -> Iterator[str]:
        """Yield every reconstructor backend in play, as each is named."""
        return self._library_parameter.choices()

    def available_algorithms(self, library: str | None = None) -> Iterator[str]:
        """Yield the algorithms *library* offers, or the selected one's when None.

        Asking about a library does not select it. Enumerating what is on
        offer is exactly what a panel does at startup, and a listing that
        selected as it went would walk the selection through every backend
        and leave it on the last.
        """
        name = self._library_parameter.get_value() if library is None else library
        return iter(self._algorithms_by_library.get(name.casefold(), ()))

    def current_selection(self) -> tuple[str, str]:
        """The selected library and algorithm, as each is named."""
        return self._algorithm_parameter.get_current_parts()

    def set_reconstructor_if_provided(
        self, *, library: str | None = None, algorithm: str | None = None
    ) -> None:
        """Select what was named, leaving alone whatever was not.

        The library goes first: the algorithm is resolved against whichever
        library is selected, so setting them the other way round would
        resolve the algorithm against the one being replaced.
        """
        if library is not None:
            self._library_parameter.set_value(library)

        if algorithm is not None:
            self._algorithm_parameter.set_value(algorithm)

    def is_reconstructor_trainable(self) -> bool:
        reconstructor = self._algorithm_parameter.get_current_reconstructor()
        return isinstance(reconstructor, TrainableReconstructor)
