from __future__ import annotations
from abc import ABC, abstractmethod
from bisect import bisect
from collections.abc import Sequence
from pathlib import Path
from typing import overload, Final
import logging
import tempfile

import h5py
import numpy
import numpy.typing

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionDataset,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPatterns,
    SimpleDiffractionDataset,
)
from ptychodus.api.tree import SimpleTreeNode
from ptychodus.api.units import BYTES_PER_MEGABYTE

from ..task_manager import BackgroundTask, TaskManager
from ._loader import ArrayAssembler, AssembledDiffractionData, LoadAllArrays, LoadArray
from .bad_pixels import BadPixelsProvider
from .settings import DiffractionSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)

__all__ = [
    'AssembledDiffractionArray',
    'AssembledDiffractionDataset',
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


class AssembledDiffractionArray(DiffractionArray):
    def __init__(
        self,
        array_index: int,
        label: str,
        data: AssembledDiffractionData,
    ) -> None:
        super().__init__()
        self._array_index = array_index
        self._label = label
        self._indexes = data.indexes
        self._patterns = data.patterns
        self._pattern_counts = data.pattern_counts
        self._average_pattern = data.patterns.mean(axis=0)

    @classmethod
    def create_null(cls) -> AssembledDiffractionArray:
        data = AssembledDiffractionData.create_null()
        return cls(0, 'null', data)

    @property
    def array_index(self) -> int:
        return self._array_index

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self) -> DiffractionPatterns:
        return self._patterns

    def get_pattern(self, index: int) -> DiffractionPatterns:
        return self._patterns[index]

    def get_pattern_counts(self, index: int) -> int:
        return self._pattern_counts[index]

    def get_mean_pattern_counts(self) -> float:
        return numpy.mean(self._pattern_counts).item()

    def get_max_pattern_counts(self) -> int:
        return max(self._pattern_counts)

    def get_average_pattern(self) -> DiffractionPatterns:
        return self._average_pattern


class AssembledDiffractionDataset(DiffractionDataset, ArrayAssembler):
    PATTERNS_KEY: Final[str] = 'patterns'
    INDEXES_KEY: Final[str] = 'indexes'
    BAD_PIXELS_KEY: Final[str] = 'bad_pixels'

    def __init__(
        self,
        settings: DiffractionSettings,
        sizer: PatternSizer,
        bad_pixels_provider: BadPixelsProvider,
        task_manager: TaskManager,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._bad_pixels_provider = bad_pixels_provider
        self._task_manager = task_manager
        self._observer_list: list[DiffractionDatasetObserver] = []

        self._dataset = SimpleDiffractionDataset.create_null()
        self._data = AssembledDiffractionData.create_null()
        self._array_list: list[AssembledDiffractionArray] = list()
        self._array_counter = 0
        self._array_loader: LoadAllArrays | None = None

    def add_observer(self, observer: DiffractionDatasetObserver) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: DiffractionDatasetObserver) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def get_layout(self) -> SimpleTreeNode:
        return self._dataset.get_layout()

    def get_metadata(self) -> DiffractionMetadata:
        return self._dataset.get_metadata()

    def get_bad_pixels(self) -> BadPixels | None:
        return self._dataset.get_bad_pixels()

    def get_assembled_indexes(self) -> DiffractionIndexes:
        return self._data.indexes[self._data.indexes >= 0]

    def get_assembled_patterns(self) -> DiffractionPatterns:
        return self._data.patterns[self._data.indexes >= 0]

    def get_maximum_pattern_counts(self) -> int:
        return self._data.pattern_counts[self._data.indexes >= 0].max()

    @overload
    def __getitem__(self, index: int) -> AssembledDiffractionArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[AssembledDiffractionArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> AssembledDiffractionArray | Sequence[AssembledDiffractionArray]:
        return self._array_list[index]

    def __len__(self) -> int:
        return len(self._array_list)

    def _create_array_loader(self, array: DiffractionArray) -> BackgroundTask:
        """Load a new array into the dataset. Assumes that arrays arrive in order."""
        bad_pixels = self._dataset.get_bad_pixels()

        if bad_pixels is None:
            raise RuntimeError('Cannot load array without bad pixel map!')

        array_index = self._array_counter
        self._array_counter += 1

        processor = self._sizer.get_processor()
        return LoadArray(array_index, array, bad_pixels, processor, self)

    def append_array(self, array: DiffractionArray) -> None:
        task = self._create_array_loader(array)
        self._task_manager.put_background_task(task)

    def _insert_array(self, array: AssembledDiffractionArray) -> None:
        pos = bisect(self._array_list, array.array_index, key=lambda x: x.array_index)
        self._array_list.insert(pos, array)

        for observer in self._observer_list:
            observer.handle_array_inserted(pos)

    def _assemble_array(
        self,
        array_index: int,
        label: str,
        data: AssembledDiffractionData,
    ) -> None:
        metadata = self.get_metadata()
        num_patterns_per_array = metadata.num_patterns_per_array
        offset = sum(num_patterns_per_array[:array_index])
        assembled_indexes = slice(offset, offset + num_patterns_per_array[array_index])

        self._data.indexes[assembled_indexes] = data.indexes
        indexes_view = self._data.indexes[assembled_indexes]
        indexes_view.flags.writeable = False

        self._data.patterns[assembled_indexes, :, :] = data.patterns
        patterns_view = self._data.patterns[assembled_indexes, :, :]
        patterns_view.flags.writeable = False

        self._data.pattern_counts[assembled_indexes] = data.pattern_counts
        pattern_counts_view = self._data.pattern_counts[assembled_indexes]
        pattern_counts_view.flags.writeable = False

        data_views = AssembledDiffractionData(
            indexes=indexes_view,
            patterns=patterns_view,
            pattern_counts=pattern_counts_view,
        )
        assembled_array = AssembledDiffractionArray(
            array_index=array_index,
            label=label,
            data=data_views,
        )

        self._task_manager.put_foreground_task(lambda: self._insert_array(assembled_array))

    def clear(self) -> None:
        self._dataset = SimpleDiffractionDataset.create_null()
        self._data = AssembledDiffractionData.create_null()
        self._array_list.clear()
        self._array_counter = 0
        self._array_loader = None

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def reload(self, dataset: DiffractionDataset, *, process_patterns: bool) -> None:
        # FIXME process_patterns
        self.clear()

        metadata = dataset.get_metadata()
        layout = dataset.get_layout()
        processor = self._sizer.get_processor()
        bad_pixels = self._bad_pixels_provider.get_bad_pixels()
        processed_bad_pixels = processor.process_bad_pixels(bad_pixels)
        self._dataset = SimpleDiffractionDataset(metadata, layout, [], processed_bad_pixels)

        num_patterns_total = sum(metadata.num_patterns_per_array)
        indexes = -numpy.ones(num_patterns_total, dtype=int)

        patterns_extent = self._sizer.get_processed_image_extent()
        patterns_shape = num_patterns_total, *patterns_extent.shape
        patterns_dtype = metadata.pattern_dtype
        pattern_counts = numpy.zeros(num_patterns_total, dtype=patterns_dtype)

        if self._settings.memmap_enabled.get_value():
            scratch_dir = self._settings.scratch_directory.get_value()
            scratch_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            npy_tmp_file = tempfile.NamedTemporaryFile(dir=scratch_dir, suffix='.npy')
            logger.info(f'Scratch data file {npy_tmp_file.name} is {patterns_shape}')
            patterns: DiffractionPatterns = numpy.memmap(
                npy_tmp_file, dtype=patterns_dtype, shape=patterns_shape
            )
            patterns[:] = 0
        else:
            logger.info(f'Scratch memory is {patterns_shape}')
            patterns = numpy.zeros(patterns_shape, dtype=patterns_dtype)
            logger.debug(f'{patterns.nbytes / BYTES_PER_MEGABYTE:.2f}MB allocated for patterns')

        self._data = AssembledDiffractionData(
            indexes=indexes,
            patterns=patterns,
            pattern_counts=pattern_counts,
        )

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

        self._array_loader = LoadAllArrays(
            dataset, processor, processed_bad_pixels, self, self._task_manager
        )

    def load_all_arrays(self) -> None:
        if self._array_loader is None:
            logger.warning('Arrays have already been loaded!')
        else:
            self._task_manager.put_background_task(self._array_loader)
            self._array_loader = None

    def import_assembled_patterns(self, file_path: Path) -> None:
        if file_path.is_file():
            self.clear()
            logger.info(f'Importing assembled dataset from "{file_path}"')

            with h5py.File(file_path, 'r') as h5_file:
                h5_indexes = h5_file[self.INDEXES_KEY]

                if not isinstance(h5_indexes, h5py.Dataset):
                    raise ValueError('Indexes are not a dataset!')

                h5_patterns = h5_file[self.PATTERNS_KEY]

                if not isinstance(h5_patterns, h5py.Dataset):
                    raise ValueError('Patterns are not a dataset!')

                h5_bad_pixels = h5_file[self.BAD_PIXELS_KEY]

                if not isinstance(h5_bad_pixels, h5py.Dataset):
                    raise ValueError('Bad pixels are not a dataset!')

                bad_pixels = h5_bad_pixels[()]
                self._bad_pixels_provider.set_bad_pixels(bad_pixels)

                self._data = AssembledDiffractionData.create_pattern_counts(
                    indexes=h5_indexes[()],
                    patterns=h5_patterns[()],  # TODO support memmap
                    bad_pixels=bad_pixels,
                )

            # FIXME deconflict detector size with bad_pixels_provider
            num_patterns, detector_height, detector_width = self._data.patterns.shape
            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=self._data.patterns.dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                file_path=file_path,
            )
            contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
            array = AssembledDiffractionArray(
                array_index=0,
                label=file_path.stem,
                data=self._data,
            )

            self._dataset = SimpleDiffractionDataset(metadata, contents_tree, [], bad_pixels)
            self._array_list = [array]
            self._array_counter = 1

            for observer in self._observer_list:
                observer.handle_dataset_reloaded()
        else:
            logger.warning(f'Refusing to read invalid file path {file_path}')

    def export_assembled_patterns(self, file_path: Path, compression: str = 'lzf') -> None:
        logger.info(f'Exporting assembled dataset to "{file_path}"')

        with h5py.File(file_path, 'w') as h5_file:
            h5_file.create_dataset(
                self.PATTERNS_KEY, data=self.get_assembled_patterns(), compression=compression
            )
            h5_file.create_dataset(
                self.INDEXES_KEY, data=self.get_assembled_indexes(), compression=compression
            )
            h5_file.create_dataset(
                self.BAD_PIXELS_KEY, data=self.get_bad_pixels(), compression=compression
            )

    def get_info_text(self) -> str:
        file_path = self.get_metadata().file_path
        return self._data.get_info_text(file_path.stem if file_path else 'None')
