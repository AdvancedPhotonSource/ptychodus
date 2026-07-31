from __future__ import annotations
from abc import ABC, abstractmethod
from bisect import bisect
from collections.abc import Sequence
from pathlib import Path
from typing import IO, overload
import logging
import tempfile

import numpy

from ptychodus.api.common import BYTES_PER_MEGABYTE
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionDataset,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPattern,
    DiffractionPatterns,
    SimpleDiffractionDataset,
)
from ptychodus.api.io import AssembledDiffractionData, load_diffraction_data, save_diffraction_data
from ptychodus.api.tree import SimpleTreeNode

from ..task_manager import BackgroundTask, TaskManager
from ._loader import ArrayAssembler, LoadAllArrays, LoadArray
from .monitor import DiffractionTaskMonitor
from .settings import DetectorSettings, DiffractionSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)


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
        self._data = data
        self._pattern_counts = data.get_pattern_counts()
        self._average_pattern = data.get_average_pattern()

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
        return self._data.get_indexes()

    def get_patterns(self) -> DiffractionPatterns:
        return self._data.get_patterns()

    def get_pattern(self, index: int) -> DiffractionPattern:
        return self._data.get_pattern(index)

    def get_pattern_counts(self, index: int) -> int:
        return self._pattern_counts[index]

    def get_mean_pattern_counts(self) -> float:
        return numpy.mean(self._pattern_counts).item()

    def get_max_pattern_counts(self) -> int:
        return self._pattern_counts.max().item()

    def get_average_pattern(self) -> DiffractionPattern:
        return self._average_pattern


class AssembledDiffractionDataset(DiffractionDataset, ArrayAssembler):
    def __init__(
        self,
        settings: DiffractionSettings,
        sizer: PatternSizer,
        detector_settings: DetectorSettings,
        task_manager: TaskManager,
        task_monitor: DiffractionTaskMonitor,
        *,
        name: str = 'default',
    ) -> None:
        super().__init__()
        self._name = name
        self._settings = settings
        self._sizer = sizer
        self._detector_settings = detector_settings
        self._task_manager = task_manager
        self._task_monitor = task_monitor
        self._observer_list: list[DiffractionDatasetObserver] = []

        self._dataset = SimpleDiffractionDataset.create_null()
        self._data = AssembledDiffractionData.create_null()
        self._array_list: list[AssembledDiffractionArray] = list()
        self._array_counter = 0
        self._array_loader: LoadAllArrays | None = None
        self._scratch_tempfile: IO[bytes] | None = None

        # Raw (pre-processing) bad-pixel mask; defaults to all-good sized to the
        # current detector extent. Overridden via set_bad_pixels() or picked up
        # from an incoming DiffractionDataset in reload().
        self._bad_pixels = self._create_default_bad_pixels()

    def _create_default_bad_pixels(self) -> BadPixels:
        height_px = self._detector_settings.height_px.get_value()
        width_px = self._detector_settings.width_px.get_value()
        return numpy.zeros((height_px, width_px), dtype=numpy.bool_)

    def get_name(self) -> str:
        return self._name

    def set_bad_pixels(self, bad_pixels: BadPixels) -> None:
        if bad_pixels.ndim != 2:
            raise ValueError(f'Bad pixels array must be 2D, got {bad_pixels.ndim}D.')
        self._bad_pixels = bad_pixels

    def reset_bad_pixels(self) -> None:
        self._bad_pixels = self._create_default_bad_pixels()

    def add_observer(self, observer: DiffractionDatasetObserver) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: DiffractionDatasetObserver) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def get_metadata(self) -> DiffractionMetadata:
        return self._dataset.get_metadata()

    def get_layout(self) -> SimpleTreeNode:
        return self._dataset.get_layout()

    def _get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self._detector_settings.pixel_width_m.get_value(),
            height_m=self._detector_settings.pixel_height_m.get_value(),
        )

    def get_bad_pixels(self) -> BadPixels:
        return self._bad_pixels

    def get_assembled_data(self) -> AssembledDiffractionData:
        return self._data

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

    def create_array_loader(
        self, array: DiffractionArray, *, process_patterns: bool
    ) -> BackgroundTask:
        """Build a loader task for one array. Loaders are assigned a monotonic array_index;
        arrays may complete out of order and are sorted on insertion via bisect."""
        array_index = self._array_counter
        self._array_counter += 1

        pipeline = self._sizer.get_prep_pipeline() if process_patterns else None
        processed_bad_pixels = (
            pipeline.apply_to_mask(self._bad_pixels) if pipeline is not None else self._bad_pixels
        )
        return LoadArray(
            array_index,
            array,
            self._get_pixel_geometry(),
            raw_bad_pixels=self._bad_pixels,
            processed_bad_pixels=processed_bad_pixels,
            pipeline=pipeline,
            assembler=self,
        )

    def append_array(self, array: DiffractionArray, *, process_patterns: bool = True) -> None:
        task = self.create_array_loader(array, process_patterns=process_patterns)
        self._task_manager.put_background_task(task)

    def _insert_array(self, array: AssembledDiffractionArray) -> None:
        pos = bisect(self._array_list, array.array_index, key=lambda x: x.array_index)
        self._array_list.insert(pos, array)

        for observer in self._observer_list:
            observer.handle_array_inserted(pos)

    def assemble_array(
        self,
        array_index: int,
        label: str,
        data: AssembledDiffractionData,
    ) -> None:
        metadata = self.get_metadata()
        num_patterns_per_array = metadata.num_patterns_per_array
        offset = sum(num_patterns_per_array[:array_index])
        assembled_data_view = self._data.assemble(data, offset)
        assembled_array = AssembledDiffractionArray(
            array_index=array_index,
            label=label,
            data=assembled_data_view,
        )
        self._task_manager.put_foreground_task(lambda: self._insert_array(assembled_array))

    def clear(self) -> None:
        self._dataset = SimpleDiffractionDataset.create_null()
        self._data = AssembledDiffractionData.create_null()
        self._array_list.clear()
        self._array_counter = 0
        self._array_loader = None
        self._bad_pixels = self._create_default_bad_pixels()

        if self._scratch_tempfile is not None:
            self._scratch_tempfile.close()
            self._scratch_tempfile = None

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def reload(self, dataset: DiffractionDataset) -> None:
        self.clear()
        self._dataset = SimpleDiffractionDataset(dataset.get_metadata(), dataset.get_layout(), [])
        self._bad_pixels = dataset.get_bad_pixels()
        self._array_loader = LoadAllArrays(dataset, self, self._task_manager, self._task_monitor)

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def load_all_arrays(self, *, process_patterns: bool, block: bool) -> None:
        if self._array_loader is None:
            logger.warning('No dataset queued for loading; call reload() first.')
            return

        metadata = self._dataset.get_metadata()

        if metadata.detector_extent is not None:
            self._detector_settings.height_px.set_value(metadata.detector_extent.height_px)
            self._detector_settings.width_px.set_value(metadata.detector_extent.width_px)

        bad_pixels = self._bad_pixels

        if process_patterns:
            pipeline = self._sizer.get_prep_pipeline()
            bad_pixels = pipeline.apply_to_mask(bad_pixels)
            self._array_loader.enable_pattern_processing()

        self._dataset = SimpleDiffractionDataset(
            metadata, self._dataset.get_layout(), [], bad_pixels
        )

        num_patterns_total = sum(metadata.num_patterns_per_array)
        indexes = -numpy.ones(num_patterns_total, dtype=int)

        patterns_shape = num_patterns_total, bad_pixels.shape[-2], bad_pixels.shape[-1]
        patterns_dtype = metadata.pattern_dtype

        if self._settings.memmap_enabled.get_value():
            scratch_dir = self._settings.scratch_directory.get_value()
            scratch_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            # Held on self so the file persists on disk for as long as the memmap is live;
            # released in clear().
            self._scratch_tempfile = tempfile.NamedTemporaryFile(dir=scratch_dir, suffix='.npy')
            logger.info(f'Scratch data file {self._scratch_tempfile.name} is {patterns_shape}')
            patterns: DiffractionPatterns = numpy.memmap(
                self._scratch_tempfile, dtype=patterns_dtype, shape=patterns_shape
            )
            patterns[:] = 0
        else:
            logger.info(f'Scratch memory is {patterns_shape}')
            patterns = numpy.zeros(patterns_shape, dtype=patterns_dtype)
            logger.debug(f'{patterns.nbytes / BYTES_PER_MEGABYTE:.2f}MB allocated for patterns')

        self._data = AssembledDiffractionData(
            indexes, patterns, self._get_pixel_geometry(), bad_pixels
        )

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

        # load all arrays in background
        finished_event = self._array_loader.get_finished_event()
        self._task_manager.put_background_task(self._array_loader)
        self._array_loader = None

        if block:
            while not self._task_manager.is_stopping:
                if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                    break

    def _generate_dataset_for_assembled_data(self, file_path: Path | None = None) -> None:
        num_patterns, detector_height, detector_width = self._data.get_patterns_shape()
        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=self._data.get_patterns_dtype(),
            detector_extent=ImageExtent(detector_width, detector_height),
            file_path=file_path,
        )
        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        array = AssembledDiffractionArray(
            array_index=0,
            label='In-Memory' if file_path is None else file_path.stem,
            data=self._data,
        )
        self._bad_pixels = self._data.get_bad_pixels()
        self._dataset = SimpleDiffractionDataset(metadata, contents_tree, [], self._bad_pixels)
        self._array_list = [array]
        self._array_counter = 1

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def set_assembled_patterns(self, data: AssembledDiffractionData) -> None:
        self.clear()
        self._data = data
        self._generate_dataset_for_assembled_data(file_path=None)

    def import_assembled_patterns(self, file_path: Path) -> None:
        if file_path.is_file():
            self.clear()
            logger.info(f'Importing assembled dataset from "{file_path}"')
            self._data = load_diffraction_data(file_path)
            self._generate_dataset_for_assembled_data(file_path=file_path)
        else:
            logger.warning(f'Refusing to read invalid file path {file_path}')

    def export_assembled_patterns(self, file_path: Path, compression: str = 'lzf') -> None:
        logger.info(f'Exporting assembled dataset to "{file_path}"')
        save_diffraction_data(file_path, self._data, compression=compression)

    def get_info_text(self) -> str:
        file_path = self.get_metadata().file_path
        label = file_path.stem if file_path else 'None'
        return f'{label}: {self._data}'
