from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload
import logging
import queue
import tempfile
import threading

import numpy
import numpy.typing

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    BooleanArrayType,
    DiffractionDataset,
    DiffractionMetadata,
    DiffractionPatternArray,
    PatternDataType,
    PatternIndexesType,
    SimpleDiffractionPatternArray,
)
from ptychodus.api.tree import SimpleTreeNode

from .settings import PatternSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)

__all__ = [
    'AssembledDiffractionDataset',
    'DiffractionDatasetObserver',
    'ObservableDiffractionDataset',
]


class DiffractionDatasetObserver(ABC):
    @abstractmethod
    def handle_array_inserted(self, index: int) -> None:
        pass

    @abstractmethod
    def handle_array_changed(self, index: int) -> None:
        pass

    @abstractmethod
    def handle_dataset_reloaded(self) -> None:
        pass


class ObservableDiffractionDataset(DiffractionDataset):
    @abstractmethod
    def add_observer(self, observer: DiffractionDatasetObserver) -> None:
        pass


@dataclass(frozen=True)
class ArrayLoaderTask:
    array: DiffractionPatternArray
    index: int


class ArrayLoader:
    def __init__(self, settings: PatternSettings, sizer: PatternSizer) -> None:
        self._settings = settings
        self._sizer = sizer
        self._input_queue: queue.Queue[ArrayLoaderTask] = queue.Queue()
        self._output_queue: queue.Queue[ArrayLoaderTask] = queue.Queue()
        self._workers: list[threading.Thread] = list()
        self._stop_work_event = threading.Event()

    @property
    def input_queue_size(self) -> int:
        return self._input_queue.qsize()

    @property
    def output_queue_size(self) -> int:
        return self._output_queue.qsize()

    def submit_task(self, task: ArrayLoaderTask) -> None:
        self._input_queue.put(task)

    def _load_arrays(self) -> None:
        processor = self._sizer.get_processor()

        while not self._stop_work_event.is_set():
            try:
                task = self._input_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                processed_array = processor(task.array)
            except Exception:
                logger.exception('Error while loading array!')
            else:
                completed_task = ArrayLoaderTask(processed_array, task.index)
                self._output_queue.put(completed_task)
            finally:
                self._input_queue.task_done()

    def completed_tasks(self) -> Iterator[ArrayLoaderTask]:
        while True:
            try:
                task = self._output_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self._output_queue.task_done()

            yield task

    def start(self) -> None:
        logger.info('Starting data loader...')
        self._stop_work_event.clear()

        # clear assembly queue
        for _ in self.completed_tasks():
            pass

        for index in range(self._settings.numberOfDataThreads.getValue()):
            thread = threading.Thread(target=self._load_arrays)
            thread.start()
            self._workers.append(thread)

        logger.info('Data loader started.')

    def stop(self, *, finish_loading: bool) -> None:
        if self._stop_work_event.is_set():
            logger.info('Data loader already stopped.')
            return

        logger.info('Stopping data loader...')

        if finish_loading:
            self._input_queue.join()
        else:
            # clear loading queue
            while True:
                try:
                    self._input_queue.get(block=False)
                except queue.Empty:
                    break
                else:
                    self._input_queue.task_done()

        self._stop_work_event.set()

        while self._workers:
            thread = self._workers.pop()
            thread.join()

        logger.info('Data loader stopped.')


class AssembledDiffractionDataset(ObservableDiffractionDataset):
    def __init__(self, settings: PatternSettings, sizer: PatternSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._loader = ArrayLoader(settings, sizer)
        self._observer_list: list[DiffractionDatasetObserver] = []

        self._contents_tree = SimpleTreeNode.createRoot([])
        self._metadata = DiffractionMetadata.createNullInstance()
        self._indexes: PatternIndexesType = numpy.zeros((), dtype=int)
        self._data: PatternDataType = numpy.zeros((0, 0, 0), dtype=int)
        self._arrays: list[DiffractionPatternArray] = list()

    @property
    def queue_size(self) -> int:
        return self._loader.input_queue_size + self._loader.output_queue_size

    def start_loading(self) -> None:
        self._loader.start()

    def finish_loading(self, *, block: bool = True) -> None:
        self._loader.stop(finish_loading=block)

    def add_observer(self, observer: DiffractionDatasetObserver) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: DiffractionDatasetObserver) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contents_tree

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def get_processed_bad_pixels(self) -> BooleanArrayType:
        # TODO support loading from file
        # TODO keep consist with processed patterns
        pattern_extent = self._sizer.get_processed_image_extent()
        return numpy.full(pattern_extent.shape, False)

    def get_assembled_indexes(self) -> PatternIndexesType:
        return self._indexes[self._indexes >= 0]

    def get_assembled_patterns(self) -> PatternDataType:
        return self._data[self._indexes >= 0]

    def get_assembled_pattern_counts(self, pattern_index: int) -> int:  # FIXME use
        pattern = self._data[pattern_index]
        good_pixels = numpy.logical_not(self.get_processed_bad_pixels())
        return numpy.sum(pattern[good_pixels])

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DiffractionPatternArray | Sequence[DiffractionPatternArray]:
        return self._arrays[index]

    def __len__(self) -> int:
        return len(self._arrays)

    def append_array(self, array: DiffractionPatternArray) -> None:
        """Load a new array into the dataset. Assumes that arrays arrive in order."""
        array_index = len(self._arrays)
        array_size = self._metadata.numberOfPatternsPerArray

        start = array_index * array_size
        end = (array_index + 1) * array_size

        pattern_indexes = self._indexes[start:end]
        pattern_indexes.flags.writeable = False

        pattern_data = self._data[start:end, :, :]
        pattern_data.flags.writeable = False

        assembled_array = SimpleDiffractionPatternArray(
            label=array.getLabel(), indexes=pattern_indexes, data=pattern_data
        )
        self._arrays.append(assembled_array)

        task = ArrayLoaderTask(array, array_index)
        self._loader.submit_task(task)

        for observer in self._observer_list:
            observer.handle_array_inserted(array_index)

    def assemble_patterns(self) -> None:
        for task in self._loader.completed_tasks():
            array_index = task.index
            array_size = self._metadata.numberOfPatternsPerArray

            start = array_index * array_size
            end = (array_index + 1) * array_size

            loaded_array = task.array
            self._indexes[start:end] = loaded_array.getIndexes()
            self._data[start:end, :, :] = loaded_array.getData()

            for observer in self._observer_list:
                observer.handle_array_changed(array_index)

    def clear(self) -> None:
        self._loader.stop(finish_loading=False)
        self._contents_tree = SimpleTreeNode.createRoot([])
        self._metadata = DiffractionMetadata.createNullInstance()
        self._indexes = numpy.zeros((), dtype=int)
        self._data = numpy.zeros((0, 0, 0), dtype=int)
        self._arrays.clear()

        for _ in self._loader.completed_tasks():
            pass

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def reload(self, dataset: DiffractionDataset) -> None:
        self.clear()
        self._contents_tree = dataset.getContentsTree()
        self._metadata = dataset.getMetadata()
        self._indexes = -numpy.ones(self._metadata.numberOfPatternsTotal, dtype=int)

        pattern_extent = self._sizer.get_processed_image_extent()
        data_shape = self._indexes.size, *pattern_extent.shape
        data_dtype = self._metadata.patternDataType

        if self._settings.memmapEnabled.getValue():
            scratch_dir = self._settings.scratchDirectory.getValue()
            scratch_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            npy_tmp_file = tempfile.NamedTemporaryFile(dir=scratch_dir, suffix='.npy')
            logger.debug(f'Scratch data file {npy_tmp_file.name} is {data_shape}')
            self._data = numpy.memmap(npy_tmp_file, dtype=data_dtype, shape=data_shape)
            self._data[:] = 0
        else:
            logger.debug(f'Scratch memory is {data_shape}')
            self._data = numpy.zeros(data_shape, dtype=data_dtype)

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

        for array in dataset:
            self.append_array(array)

    def import_assembled_patterns(self, filePath: Path) -> None:
        if filePath.is_file():
            self.clear()
            logger.debug(f'Reading processed patterns from "{filePath}"')

            try:
                contents = numpy.load(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc

            self._indexes = contents['indexes']
            self._data = contents['patterns']
            numberOfPatterns, detectorHeight, detectorWidth = self._data.shape

            self._contents_tree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
            self._metadata = DiffractionMetadata(
                numberOfPatternsPerArray=numberOfPatterns,
                numberOfPatternsTotal=numberOfPatterns,
                patternDataType=self._data.dtype,
                detectorExtent=ImageExtent(detectorWidth, detectorHeight),
            )
            self._arrays = [
                SimpleDiffractionPatternArray(
                    label='Imported',
                    indexes=self._indexes,
                    data=self._data,
                )
            ]

            for observer in self._observer_list:
                observer.handle_dataset_reloaded()
        else:
            logger.warning(f'Refusing to read invalid file path {filePath}')

    def export_assembled_patterns(self, filePath: Path) -> None:
        logger.debug(f'Writing processed patterns to "{filePath}"')
        numpy.savez(
            filePath,
            indexes=self.get_assembled_indexes(),
            patterns=self.get_assembled_patterns(),
        )

    def get_info_text(self) -> str:
        file_path = self._metadata.filePath
        label = file_path.stem if file_path else 'None'
        number, height, width = self._data.shape
        dtype = str(self._data.dtype)
        sizeInMB = self._data.nbytes / (1024 * 1024)
        return f'{label}: {number} x {width}W x {height}H {dtype} [{sizeInMB:.2f}MB]'
