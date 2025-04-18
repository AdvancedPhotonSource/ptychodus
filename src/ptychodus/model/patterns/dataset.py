from __future__ import annotations
from abc import ABC, abstractmethod
from bisect import bisect
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
    DiffractionDataset,
    DiffractionMetadata,
    DiffractionPatternArray,
    PatternDataType,
    PatternIndexesType,
)
from ptychodus.api.tree import SimpleTreeNode
from ptychodus.api.typing import BooleanArrayType
from ptychodus.api.units import BYTES_PER_MEGABYTE

from .settings import PatternSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)

__all__ = [
    'AssembledDiffractionDataset',
    'AssembledDiffractionPatternArray',
    'DiffractionDatasetObserver',
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


class AssembledDiffractionPatternArray(DiffractionPatternArray):
    def __init__(
        self,
        label: str,
        indexes: PatternIndexesType,
        data: PatternDataType,
        good_pixels: BooleanArrayType,
        array_index: int,
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._data = data
        self._good_pixels = good_pixels
        self._array_index = array_index

    @classmethod
    def create_null(cls) -> AssembledDiffractionPatternArray:
        indexes = numpy.array([0])
        data = numpy.zeros((1, 1, 1), dtype=numpy.uint16)
        good_pixels = numpy.full((1, 1), True)
        return cls('null', indexes, data, good_pixels, 0)

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> PatternIndexesType:
        return self._indexes

    def get_data(self) -> PatternDataType:
        return self._data

    def get_pattern(self, index: int) -> PatternDataType:
        return self._data[index]

    def get_pattern_counts(self, index: int) -> int:
        pattern = self._data[index]
        return pattern[self._good_pixels].sum()

    def get_average_pattern(self) -> PatternDataType:
        return self._data.mean(axis=0)

    def get_mean_pattern_counts(self) -> float:
        loaded_data = self._data[self._indexes >= 0]
        total_counts = numpy.sum(loaded_data[:, self._good_pixels], axis=-1)
        return total_counts.mean()

    def get_max_pattern_counts(self) -> int:
        loaded_data = self._data[self._indexes >= 0]
        total_counts = numpy.sum(loaded_data[:, self._good_pixels], axis=-1)
        return total_counts.max()

    def get_array_index(self) -> int:
        return self._array_index


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
            except FileNotFoundError:
                logger.warning(f'File not found for array index={task.index}.')
            except OSError:
                logger.error(f'OS error while reading array index={task.index}.')
            except Exception:
                logger.exception(f'Error while loading array index={task.index}!')
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

        for index in range(self._settings.num_data_threads.get_value()):
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


class AssembledDiffractionDataset(DiffractionDataset):
    def __init__(self, settings: PatternSettings, sizer: PatternSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._loader = ArrayLoader(settings, sizer)
        self._observer_list: list[DiffractionDatasetObserver] = []

        self._contents_tree = SimpleTreeNode.create_root([])
        self._metadata = DiffractionMetadata.create_null()
        self._indexes: PatternIndexesType = numpy.zeros((), dtype=int)
        self._data: PatternDataType = numpy.zeros((0, 0, 0), dtype=int)
        self._arrays: list[AssembledDiffractionPatternArray] = list()
        self._array_counter = 0

    @property
    def queue_size(self) -> int:
        return self._loader.input_queue_size + self._loader.output_queue_size

    def start_loading(self) -> None:
        pattern_extent = self._sizer.get_processed_image_extent()
        data_shape = self._indexes.size, *pattern_extent.shape
        data_dtype = self._metadata.pattern_dtype

        if self._settings.is_memmap_enabled.get_value():
            scratch_dir = self._settings.scratch_directory.get_value()
            scratch_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            npy_tmp_file = tempfile.NamedTemporaryFile(dir=scratch_dir, suffix='.npy')
            logger.info(f'Scratch data file {npy_tmp_file.name} is {data_shape}')
            self._data = numpy.memmap(npy_tmp_file, dtype=data_dtype, shape=data_shape)
            self._data[:] = 0
        else:
            logger.info(f'Scratch memory is {data_shape}')
            self._data = numpy.zeros(data_shape, dtype=data_dtype)

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

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

    def get_contents_tree(self) -> SimpleTreeNode:
        return self._contents_tree

    def get_metadata(self) -> DiffractionMetadata:
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

    def get_maximum_pattern_counts(self) -> int:
        patterns = self.get_assembled_patterns()
        good_pixels = numpy.logical_not(self.get_processed_bad_pixels())
        try:
            total_counts = numpy.sum(patterns[:, good_pixels], axis=-1)
        except IndexError:
            # patterns not loaded
            return 0

        return total_counts.max()

    @overload
    def __getitem__(self, index: int) -> AssembledDiffractionPatternArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[AssembledDiffractionPatternArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> AssembledDiffractionPatternArray | Sequence[AssembledDiffractionPatternArray]:
        return self._arrays[index]

    def __len__(self) -> int:
        return len(self._arrays)

    def append_array(self, array: DiffractionPatternArray) -> None:
        """Load a new array into the dataset. Assumes that arrays arrive in order."""
        task = ArrayLoaderTask(array, int(self._array_counter))
        self._array_counter += 1
        self._loader.submit_task(task)

    def assemble_patterns(self) -> None:
        for task in self._loader.completed_tasks():
            array_size = self._metadata.num_patterns_per_array
            array_slice = slice(task.index * array_size, (task.index + 1) * array_size)

            self._indexes[array_slice] = task.array.get_indexes()
            pattern_indexes = self._indexes[array_slice]
            pattern_indexes.flags.writeable = False

            self._data[array_slice, :, :] = task.array.get_data()
            pattern_data = self._data[array_slice, :, :]
            pattern_data.flags.writeable = False

            array = AssembledDiffractionPatternArray(
                label=task.array.get_label(),
                indexes=pattern_indexes,
                data=pattern_data,
                good_pixels=numpy.logical_not(self.get_processed_bad_pixels()),
                array_index=task.index,
            )

            pos = bisect(self._arrays, array.get_array_index(), key=lambda x: x.get_array_index())
            self._arrays.insert(pos, array)

            for observer in self._observer_list:
                observer.handle_array_inserted(pos)

    def clear(self) -> None:
        self._loader.stop(finish_loading=False)
        self._contents_tree = SimpleTreeNode.create_root([])
        self._metadata = DiffractionMetadata.create_null()
        self._indexes = numpy.zeros((), dtype=int)
        self._data = numpy.zeros((0, 0, 0), dtype=int)
        self._arrays.clear()
        self._array_counter = 0

        for _ in self._loader.completed_tasks():
            pass

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def reload(self, dataset: DiffractionDataset) -> None:
        self.clear()
        self._contents_tree = dataset.get_contents_tree()
        self._metadata = dataset.get_metadata()
        self._indexes = -numpy.ones(self._metadata.num_patterns_total, dtype=int)

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

        for array in dataset:
            self.append_array(array)

    def import_assembled_patterns(self, file_path: Path) -> None:
        if file_path.is_file():
            self.clear()
            logger.debug(f'Reading processed patterns from "{file_path}"')

            try:
                contents = numpy.load(file_path)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{file_path}"') from exc

            self._indexes = contents['indexes']
            self._data = contents['patterns']
            num_patterns, detector_height, detector_width = self._data.shape

            self._contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
            self._metadata = DiffractionMetadata(
                num_patterns_per_array=num_patterns,
                num_patterns_total=num_patterns,
                pattern_dtype=self._data.dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
            )
            self._arrays = [
                AssembledDiffractionPatternArray(
                    label='Imported',
                    indexes=self._indexes,
                    data=self._data,
                    good_pixels=numpy.logical_not(self.get_processed_bad_pixels()),
                    array_index=0,
                )
            ]
            self._array_counter = 1

            for observer in self._observer_list:
                observer.handle_dataset_reloaded()
        else:
            logger.warning(f'Refusing to read invalid file path {file_path}')

    def export_assembled_patterns(self, file_path: Path) -> None:
        logger.debug(f'Writing processed patterns to "{file_path}"')
        numpy.savez(
            file_path,
            indexes=self.get_assembled_indexes(),
            patterns=self.get_assembled_patterns(),
        )

    def get_info_text(self) -> str:
        file_path = self._metadata.file_path
        label = file_path.stem if file_path else 'None'
        number, height, width = self._data.shape
        dtype = str(self._data.dtype)
        size_MB = self._data.nbytes / BYTES_PER_MEGABYTE  # noqa: N806
        return f'{label}: {number} x {width}W x {height}H {dtype} [{size_MB:.2f}MB]'
