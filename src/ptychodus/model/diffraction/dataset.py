from __future__ import annotations
from abc import ABC, abstractmethod
from bisect import bisect
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import IO, overload
import logging
import tempfile
import threading

import numpy

from ptychodus.api.constants import format_bytes
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPattern,
    DiffractionPatterns,
    SimpleDiffractionDataset,
)
from ptychodus.api.io import AssembledDiffractionData, load_diffraction_data, save_diffraction_data

from ..task_manager import BackgroundTask, TaskManager
from ._loader import ArrayAssembler, LoadAllArrays, LoadArray
from .monitor import DiffractionTaskMonitor
from .settings import DetectorSettings, DiffractionSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)


class DiffractionDatasetState(Enum):
    """Load state of an AssembledDiffractionDataset.

    Mirrors ProductState so the three repository items expose the same vocabulary.
    Derived from the loader handles rather than stored, so there is no extra
    invariant to keep in sync.
    """

    READY = 'ready'
    PENDING = 'pending'
    FAILED = 'failed'


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

    @abstractmethod
    def handle_pixel_geometry_changed(self) -> None:
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
        # Retained after dispatch so callers can detect \"still loading\" and read
        # any error the loader stored (see is_load_in_progress / get_last_load_error).
        self._last_array_loader: LoadAllArrays | None = None
        self._scratch_tempfile: IO[bytes] | None = None

        # Raw (pre-processing) bad-pixel mask; starts as an empty (0, 0) placeholder
        # and is always overwritten by reload() or by load_all_arrays() before any
        # array is processed, so the shape here only matters when nothing is loaded.
        self._bad_pixels = self._create_default_bad_pixels()

        # Per-dataset override for the raw detector pixel geometry. When set, takes
        # priority over metadata and DetectorSettings in get_raw_pixel_geometry().
        # Cleared by clear()/reload() so freshly-read metadata is the new source of truth.
        self._pixel_geometry_override: PixelGeometry | None = None

    def _create_default_bad_pixels(self) -> BadPixels:
        extent = self._dataset.get_metadata().detector_extent
        return numpy.zeros((extent.height_px, extent.width_px), dtype=numpy.bool_)

    def get_name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        """Set the dataset's display name. Callers must ensure uniqueness themselves
        (typically by routing the candidate through DiffractionDatasetRepository.create_unique_name).
        """
        self._name = name

    def sync_pixel_geometry_to_settings(self) -> None:
        """Promote this dataset's effective raw pixel geometry to the global fallback.

        Writes the current raw geometry (override > metadata > current fallback) into
        DetectorSettings.pixel_width_m / pixel_height_m so freshly loaded datasets that
        lack pixel metadata pick it up. Leaves this dataset's override in place.
        """
        geometry = self.get_raw_pixel_geometry()
        self._detector_settings.pixel_width_m.set_value(geometry.width_m)
        self._detector_settings.pixel_height_m.set_value(geometry.height_m)

    def set_bad_pixels(self, bad_pixels: BadPixels) -> None:
        if bad_pixels.ndim != 2:
            raise ValueError(f'Bad pixels array must be 2D, got {bad_pixels.ndim}D.')

        extent = self._dataset.get_metadata().detector_extent

        if bad_pixels.shape != extent.get_shape():
            raise ValueError(
                f'Bad pixels shape {bad_pixels.shape} does not match '
                f'loaded detector extent {extent.get_shape()}.'
            )

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

    def get_layout(self) -> DiffractionDatasetLayoutNode:
        return self._dataset.get_layout()

    def get_raw_pixel_geometry(self) -> PixelGeometry:
        """Resolve the raw (pre-processing) detector pixel geometry for this dataset.

        Priority: user override > metadata > global DetectorSettings fallback.
        """
        if self._pixel_geometry_override is not None:
            return self._pixel_geometry_override

        metadata_geometry = self._dataset.get_metadata().detector_pixel_geometry
        if metadata_geometry is not None:
            return metadata_geometry

        return PixelGeometry(
            width_m=self._detector_settings.pixel_width_m.get_value(),
            height_m=self._detector_settings.pixel_height_m.get_value(),
        )

    def set_pixel_geometry_override(self, geometry: PixelGeometry | None) -> None:
        """Set (or clear when None) the per-dataset raw pixel geometry override.

        Also mutates the assembled-data snapshot so consumers reading
        AssembledDiffractionData.get_pixel_geometry() see the update, and notifies
        observers so downstream views (tree columns, bound products) can refresh.
        """
        self._pixel_geometry_override = geometry
        self._data.set_pixel_geometry(self.get_raw_pixel_geometry())

        for observer in self._observer_list:
            observer.handle_pixel_geometry_changed()

    def get_bad_pixels(self) -> BadPixels:
        return self._bad_pixels

    def get_assembled_data(self) -> AssembledDiffractionData:
        return self._data

    def get_nbytes(self) -> int:
        return self._data.nbytes

    def is_load_in_progress(self) -> bool:
        """True while a LoadAllArrays task is queued but has not finished.

        Note: this does not track per-array append_array() streams, which are
        used only by the pvapy streaming path; products are created there only
        after the stream stops, so the streaming case doesn't rely on this.
        """
        if self._array_loader is not None:
            return True

        loader = self._last_array_loader
        return loader is not None and not loader.get_finished_event().is_set()

    def get_state(self) -> DiffractionDatasetState:
        if self.is_load_in_progress():
            return DiffractionDatasetState.PENDING

        if self.get_last_load_error() is not None:
            return DiffractionDatasetState.FAILED

        return DiffractionDatasetState.READY

    def is_pending(self) -> bool:
        return self.get_state() is DiffractionDatasetState.PENDING

    def is_failed(self) -> bool:
        return self.get_state() is DiffractionDatasetState.FAILED

    def get_last_load_error(self) -> BaseException | None:
        """First exception raised by the most recent LoadAllArrays run, or None."""
        loader = self._last_array_loader
        return loader.get_error() if loader is not None else None

    def get_last_load_finished_event(self) -> threading.Event | None:
        """The finished_event of the most recent LoadAllArrays task, if any."""
        loader = self._last_array_loader
        return loader.get_finished_event() if loader is not None else None

    def get_average_pattern(self) -> DiffractionPattern | None:
        if not self._array_list:
            return None
        weights = numpy.array(
            [array.get_num_patterns() for array in self._array_list], dtype=numpy.float64
        )
        averages = numpy.stack([array.get_average_pattern() for array in self._array_list])
        return numpy.average(averages, axis=0, weights=weights)

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

        detector_extent = self._dataset.get_metadata().detector_extent
        pipeline = self._sizer.get_prep_pipeline(detector_extent) if process_patterns else None
        processed_bad_pixels = (
            pipeline.apply_to_mask(self._bad_pixels) if pipeline is not None else self._bad_pixels
        )
        return LoadArray(
            array_index,
            array,
            self.get_raw_pixel_geometry(),
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
        self._last_array_loader = None
        self._bad_pixels = self._create_default_bad_pixels()
        self._pixel_geometry_override = None

        if self._scratch_tempfile is not None:
            self._scratch_tempfile.close()
            self._scratch_tempfile = None

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def reload(self, dataset: DiffractionDataset) -> None:
        self.clear()
        metadata = dataset.get_metadata()
        self._dataset = SimpleDiffractionDataset(metadata, dataset.get_layout(), [])
        self._bad_pixels = dataset.get_bad_pixels()
        self._array_loader = LoadAllArrays(dataset, self, self._task_manager, self._task_monitor)

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def load_all_arrays(self, *, process_patterns: bool, block: bool) -> None:
        if self._array_loader is None:
            logger.warning('No dataset queued for loading; call reload() first.')
            return

        metadata = self._dataset.get_metadata()

        bad_pixels = self._bad_pixels

        if process_patterns:
            pipeline = self._sizer.get_prep_pipeline(metadata.detector_extent)
            bad_pixels = pipeline.apply_to_mask(bad_pixels)
            self._array_loader.enable_pattern_processing()

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
            logger.debug(f'{format_bytes(patterns.nbytes)} allocated for patterns')

        self._data = AssembledDiffractionData(
            indexes, patterns, self.get_raw_pixel_geometry(), bad_pixels
        )

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

        # load all arrays in background
        loader = self._array_loader
        finished_event = loader.get_finished_event()
        self._last_array_loader = loader
        self._task_manager.put_background_task(loader)
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
        contents_tree = DiffractionDatasetLayoutNode.create_root()
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
