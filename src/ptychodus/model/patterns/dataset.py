from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
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
    PatternState,
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


class AssembledDiffractionPatternArray(DiffractionPatternArray):
    def __init__(
        self,
        label: str,
        indexes: PatternIndexesType,
        data: PatternDataType,
        start: int,
        end: int,
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._data = data
        self._start = start
        self._end = end

    def getLabel(self) -> str:
        return self._label

    def getIndexes(self) -> PatternIndexesType:
        return self._indexes[self._start : self._end]

    def getData(self) -> PatternDataType:
        data = self._data[self._start : self._end, :, :]
        data.flags.writeable = False
        return data

    def getState(self) -> PatternState:
        loaded = any(self.getIndexes() >= 0)
        return PatternState.LOADED if loaded else PatternState.UNKNOWN


class DiffractionPatternArrayLoader:
    def __init__(self, settings: PatternSettings, sizer: PatternSizer) -> None:
        self._settings = settings
        self._sizer = sizer
        self._processing_queue: queue.Queue[DiffractionPatternArray] = queue.Queue()
        self._assembly_queue: queue.Queue[DiffractionPatternArray] = queue.Queue()
        self._workers: list[threading.Thread] = list()
        self._stop_work_event = threading.Event()

    @property
    def processing_queue_size(self) -> int:
        return self._processing_queue.qsize()

    @property
    def assembly_queue_size(self) -> int:
        return self._assembly_queue.qsize()

    def load_array(self, array: DiffractionPatternArray) -> None:
        self._processing_queue.put(array)

    def _load_arrays(self) -> None:
        processor = self._sizer.get_processor()

        while not self._stop_work_event.is_set():
            try:
                array = self._processing_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                processed_array = processor(array)
            except Exception:
                logger.exception('Error while processing array!')
            else:
                self._assembly_queue.put(processed_array)
            finally:
                self._processing_queue.task_done()

    def loaded_arrays(self) -> Iterator[DiffractionPatternArray]:
        while True:
            try:
                array = self._assembly_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self._assembly_queue.task_done()

            yield array

    def start(self) -> None:
        logger.info('Starting data loader...')
        self._stop_work_event.clear()

        # clear assembly queue
        for _ in self.loaded_arrays():
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
            self._processing_queue.join()
        else:
            # clear processing queue
            while True:
                try:
                    self._processing_queue.get(block=False)
                except queue.Empty:
                    break
                else:
                    self._processing_queue.task_done()

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
        self._loader = DiffractionPatternArrayLoader(settings, sizer)
        self._observer_list: list[DiffractionDatasetObserver] = []

        self._contents_tree = SimpleTreeNode.createRoot([])
        self._metadata = DiffractionMetadata.createNullInstance()
        self._indexes: PatternIndexesType = numpy.zeros((), dtype=int)
        self._data: PatternDataType = numpy.zeros((0, 0, 0), dtype=int)
        self._arrays: list[DiffractionPatternArray] = list()

    @property
    def queue_size(self) -> int:
        return self._loader.processing_queue_size + self._loader.assembly_queue_size

    def start_processing(self) -> None:
        self._loader.start()

    def finish_processing(self, *, block: bool = True) -> None:
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

    def get_assembled_pattern_counts(self, index: int) -> int:  # FIXME use
        pattern = self._data[index]
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
        """Load a new array into the dataset."""
        self._loader.load_array(array)

        # assumes that arrays arrive in order
        array_index = len(self._arrays)
        array_size = self._metadata.numberOfPatternsPerArray
        assembled_array = AssembledDiffractionPatternArray(
            label=array.getLabel(),
            indexes=self._indexes,
            data=self._data,
            start=array_index * array_size,
            end=(array_index + 1) * array_size,
        )
        # FIXME insert sorted by index?
        self._arrays.append(assembled_array)

        for observer in self._observer_list:
            observer.handle_array_inserted(array_index)

    def assemble_patterns(self) -> None:
        for array in self._loader.loaded_arrays():
            # TODO update self._indexes
            # TODO update self._data
            # TODO record total intensity
            index = 0  # FIXME assemble

            for observer in self._observer_list:
                observer.handle_array_changed(index)

    def clear(self) -> None:
        self._loader.stop(finish_loading=False)
        self._contents_tree = SimpleTreeNode.createRoot([])
        self._metadata = DiffractionMetadata.createNullInstance()
        self._indexes = numpy.zeros((), dtype=int)
        self._data = numpy.zeros((0, 0, 0), dtype=int)
        self._arrays.clear()

        for _ in self._loader.loaded_arrays():
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
                AssembledDiffractionPatternArray(
                    label='Imported',
                    indexes=self._indexes,
                    data=self._data,
                    start=0,
                    end=numberOfPatterns,
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
