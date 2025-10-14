from __future__ import annotations
from abc import ABC, abstractmethod
from bisect import bisect
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, Final
import logging
import queue
import tempfile
import threading

import h5py
import numpy
import numpy.typing

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionDataset,
    DiffractionMetadata,
    DiffractionArray,
    DiffractionPatterns,
    DiffractionIndexes,
)
from ptychodus.api.tree import SimpleTreeNode
from ptychodus.api.units import BYTES_PER_MEGABYTE

from .settings import DiffractionSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)

__all__ = [
    'AssembledDiffractionDataset',
    'AssembledDiffractionPatternArray',
    'DiffractionDatasetObserver',
]


class DiffractionDatasetObserver(ABC):
    @abstractmethod
    def handle_bad_pixels_changed(self, num_bad_pixels: int) -> None:
        pass

    @abstractmethod
    def handle_array_inserted(self, index: int) -> None:
        pass

    @abstractmethod
    def handle_array_changed(self, index: int) -> None:
        pass

    @abstractmethod
    def handle_dataset_reloaded(self) -> None:
        pass


class AssembledDiffractionPatternArray(DiffractionArray):
    def __init__(
        self,
        label: str,
        indexes: DiffractionIndexes,
        data: DiffractionPatterns,
        bad_pixels: BadPixels,
        array_index: int,
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._data = data
        self._good_pixels = numpy.logical_not(bad_pixels)
        self._array_index = array_index

        self._average_pattern = data.mean(axis=0)
        loaded_data = data[indexes >= 0]
        self._total_counts = numpy.sum(loaded_data[:, self._good_pixels], axis=-1)

    @classmethod
    def create_null(cls) -> AssembledDiffractionPatternArray:
        indexes = numpy.array([0])
        data = numpy.zeros((1, 1, 1), dtype=numpy.uint16)
        bad_pixels = numpy.full((1, 1), False)
        return cls('null', indexes, data, bad_pixels, 0)

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self) -> DiffractionPatterns:
        return self._data

    def get_pattern(self, index: int) -> DiffractionPatterns:
        return self._data[index]

    def get_pattern_counts(self, index: int) -> int:
        pattern = self._data[index]
        return pattern[self._good_pixels].sum()

    def get_average_pattern(self) -> DiffractionPatterns:
        return self._average_pattern

    def get_mean_pattern_counts(self) -> float:
        return self._total_counts.mean()

    def get_max_pattern_counts(self) -> int:
        return self._total_counts.max()

    def get_array_index(self) -> int:
        return self._array_index


@dataclass(frozen=True)
class ArrayLoaderTask:
    array: DiffractionArray
    index: int


class ArrayLoader:
    def __init__(self, settings: DiffractionSettings, sizer: PatternSizer) -> None:
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
    PATTERNS_KEY: Final[str] = 'patterns'
    INDEXES_KEY: Final[str] = 'indexes'
    BAD_PIXELS_KEY: Final[str] = 'bad_pixels'

    def __init__(self, settings: DiffractionSettings, sizer: PatternSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._loader = ArrayLoader(settings, sizer)
        self._observer_list: list[DiffractionDatasetObserver] = []

        self._contents_tree = SimpleTreeNode.create_root([])
        self._metadata = DiffractionMetadata.create_null()
        self._indexes: DiffractionIndexes = numpy.zeros((), dtype=int)
        self._data: DiffractionPatterns = numpy.zeros((0, 0, 0), dtype=int)
        self._arrays: list[AssembledDiffractionPatternArray] = list()
        self._array_counter = 0
        self._bad_pixels: BadPixels | None = None

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
            logger.debug(f'{self._data.nbytes / BYTES_PER_MEGABYTE:.2f}MB allocated for patterns')

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

    def get_layout(self) -> SimpleTreeNode:
        return self._contents_tree

    def get_metadata(self) -> DiffractionMetadata:
        return self._metadata

    def set_bad_pixels(self, bad_pixels: BadPixels | None) -> None:
        if bad_pixels is not None:
            if bad_pixels.ndim != 2:
                raise ValueError(f'Bad pixels array must be 2D, got {bad_pixels.ndim}D.')

            actual_extent = ImageExtent(
                width_px=bad_pixels.shape[-1], height_px=bad_pixels.shape[-2]
            )
            expected_extent = self._sizer.get_detector_extent()

            if actual_extent != expected_extent:
                raise ValueError(f'Shape mismatch: {actual_extent=} {expected_extent=}')

        self._bad_pixels = bad_pixels

        num_bad_pixels = (
            0 if self._bad_pixels is None else numpy.count_nonzero(self._bad_pixels).item()
        )

        for observer in self._observer_list:
            observer.handle_bad_pixels_changed(num_bad_pixels)

    def get_bad_pixels(self) -> BadPixels | None:
        return self._bad_pixels

    def get_processed_bad_pixels(self) -> BadPixels:
        processor = self._sizer.get_processor()
        detector_extent = self._sizer.get_detector_extent()
        bad_pixels = (
            numpy.full(detector_extent.shape, False)
            if self._bad_pixels is None
            else self._bad_pixels
        )
        return processor.process_bad_pixels(bad_pixels)

    def get_assembled_indexes(self) -> DiffractionIndexes:
        return self._indexes[self._indexes >= 0]

    def get_assembled_patterns(self) -> DiffractionPatterns:
        return self._data[self._indexes >= 0]

    def get_maximum_pattern_counts(self) -> int:
        try:
            return max(array.get_max_pattern_counts() for array in self._arrays)
        except ValueError:
            # no arrays loaded
            return 0

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

    def append_array(self, array: DiffractionArray) -> None:
        """Load a new array into the dataset. Assumes that arrays arrive in order."""
        task = ArrayLoaderTask(array, self._array_counter)
        self._array_counter += 1
        self._loader.submit_task(task)

    def assemble_patterns(self) -> None:
        for task in self._loader.completed_tasks():
            offset = sum(self._metadata.num_patterns_per_array[: task.index])
            array_slice = slice(offset, offset + self._metadata.num_patterns_per_array[task.index])

            self._indexes[array_slice] = task.array.get_indexes()
            pattern_indexes = self._indexes[array_slice]
            pattern_indexes.flags.writeable = False
            task_array_data = task.array.get_patterns()

            try:
                self._data[array_slice, :, :] = task_array_data
            except ValueError as exc:
                logger.exception(exc)
                return

            pattern_data = self._data[array_slice, :, :]
            pattern_data.flags.writeable = False

            array = AssembledDiffractionPatternArray(
                label=task.array.get_label(),
                indexes=pattern_indexes,
                data=pattern_data,
                bad_pixels=self.get_processed_bad_pixels(),
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
        self._contents_tree = dataset.get_layout()
        self._metadata = dataset.get_metadata()
        self._indexes = -numpy.ones(sum(self._metadata.num_patterns_per_array), dtype=int)

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

        for array in dataset:
            self.append_array(array)

    def import_assembled_patterns(self, file_path: Path) -> None:
        if file_path.is_file():
            self.clear()
            logger.info(f'Importing assembled dataset from "{file_path}"')

            with h5py.File(file_path, 'r') as h5_file:
                h5_indexes = h5_file[self.INDEXES_KEY]

                if isinstance(h5_indexes, h5py.Dataset):
                    self._indexes = h5_indexes[()]
                else:
                    raise ValueError('Indexes are not a dataset!')

                h5_patterns = h5_file[self.PATTERNS_KEY]

                if isinstance(h5_patterns, h5py.Dataset):
                    self._data = h5_patterns[()]
                else:
                    raise ValueError('Patterns are not a dataset!')

                self._bad_pixels = None

                if self.BAD_PIXELS_KEY in h5_file:
                    h5_bad_pixels = h5_file[self.BAD_PIXELS_KEY]

                    if isinstance(h5_bad_pixels, h5py.Dataset):
                        self._bad_pixels = h5_bad_pixels[()]

            num_patterns, detector_height, detector_width = self._data.shape

            self._contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
            self._metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=self._data.dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                file_path=file_path,
            )
            self._arrays = [
                AssembledDiffractionPatternArray(
                    label=file_path.stem,
                    indexes=self._indexes,
                    data=self._data,
                    bad_pixels=self.get_processed_bad_pixels(),
                    array_index=0,
                )
            ]
            self._array_counter = 1

            for observer in self._observer_list:
                observer.handle_dataset_reloaded()
        else:
            logger.warning(f'Refusing to read invalid file path {file_path}')

    def export_assembled_patterns(self, file_path: Path) -> None:
        logger.info(f'Exporting assembled dataset to "{file_path}"')

        with h5py.File(file_path, 'w') as h5_file:
            h5_file.create_dataset(
                self.PATTERNS_KEY, data=self.get_assembled_patterns(), compression='gzip'
            )
            h5_file.create_dataset(
                self.INDEXES_KEY, data=self.get_assembled_indexes(), compression='gzip'
            )

            if self._bad_pixels is not None:
                h5_file.create_dataset(
                    self.BAD_PIXELS_KEY, data=self._bad_pixels, compression='gzip'
                )

    def get_info_text(self) -> str:
        file_path = self._metadata.file_path
        label = file_path.stem if file_path else 'None'
        number, height, width = self._data.shape
        dtype = str(self._data.dtype)
        size_MB = self._data.nbytes / BYTES_PER_MEGABYTE  # noqa: N806
        return f'{label}: {number} x {width}W x {height}H {dtype} [{size_MB:.2f}MB]'
